from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def create_multi_volume(module, array, single=False):
    """Create Volume"""
    volfact = {}
    changed = True
    api_version = array._list_available_rest_versions()
    bw_qos_size = iops_qos_size = 0
    names = []
    if '/' in module.params['name'] and (not check_vgroup(module, array)):
        module.fail_json(msg='Multi-volume create failed. Volume Group {0} does not exist.'.format(module.params['name'].split('/')[0]))
    if '::' in module.params['name']:
        if not check_pod(module, array):
            module.fail_json(msg='Multi-volume create failed. Pod {0} does not exist'.format(module.params['name'].split(':')[0]))
        pod_name = module.params['name'].split('::')[0]
        if PROMOTE_API_VERSION in api_version:
            if array.get_pod(pod_name)['promotion_status'] == 'demoted':
                module.fail_json(msg='Volume cannot be created in a demoted pod')
    array = get_array(module)
    if not single:
        for vol_num in range(module.params['start'], module.params['count'] + module.params['start']):
            names.append(module.params['name'] + str(vol_num).zfill(module.params['digits']) + module.params['suffix'])
    else:
        names.append(module.params['name'])
    if module.params['bw_qos']:
        bw_qos = int(human_to_bytes(module.params['bw_qos']))
        if bw_qos in range(1048576, 549755813888):
            bw_qos_size = bw_qos
        else:
            module.fail_json(msg='Bandwidth QoS value out of range.')
    if module.params['iops_qos']:
        iops_qos = int(human_to_real(module.params['iops_qos']))
        if iops_qos in range(100, 100000000):
            iops_qos_size = iops_qos
        else:
            module.fail_json(msg='IOPs QoS value out of range.')
    if bw_qos_size != 0 and iops_qos_size != 0:
        vols = flasharray.VolumePost(provisioned=human_to_bytes(module.params['size']), qos=flasharray.Qos(bandwidth_limit=bw_qos_size, iops_limit=iops_qos_size), subtype='regular')
    elif bw_qos_size == 0 and iops_qos_size == 0:
        vols = flasharray.VolumePost(provisioned=human_to_bytes(module.params['size']), subtype='regular')
    elif bw_qos_size == 0 and iops_qos_size != 0:
        vols = flasharray.VolumePost(provisioned=human_to_bytes(module.params['size']), qos=flasharray.Qos(iops_limit=iops_qos_size), subtype='regular')
    elif bw_qos_size != 0 and iops_qos_size == 0:
        vols = flasharray.VolumePost(provisioned=human_to_bytes(module.params['size']), qos=flasharray.Qos(bandwidth_limit=bw_qos_size), subtype='regular')
    if not module.check_mode:
        if DEFAULT_API_VERSION in api_version:
            if module.params['add_to_pgs']:
                add_to_pgs = []
                for add_pg in range(0, len(module.params['add_to_pgs'])):
                    add_to_pgs.append(flasharray.FixedReference(name=module.params['add_to_pgs'][add_pg]))
                res = array.post_volumes(names=names, volume=vols, with_default_protection=module.params['with_default_protection'], add_to_protection_groups=add_to_pgs)
            else:
                res = array.post_volumes(names=names, volume=vols, with_default_protection=module.params['with_default_protection'])
        else:
            res = array.post_volumes(names=names, volume=vols)
        if res.status_code != 200:
            module.fail_json(msg='Multi-Volume {0}#{1} creation failed: {2}'.format(module.params['name'], module.params['suffix'], res.errors[0].message))
        else:
            if VOLUME_PROMOTION_API_VERSION in api_version and module.params['promotion_state']:
                volume = flasharray.VolumePatch(requested_promotion_state=module.params['promotion_state'])
                prom_res = array.patch_volumes(names=names, volume=volume)
                if prom_res.status_code != 200:
                    array.patch_volumes(names=names, volume=flasharray.VolumePatch(destroyed=True))
                    array.delete_volumes(names=names)
                    module.warn('Failed to set promotion status on volumes. Error: {0}'.format(prom_res.errors[0].message))
            if PRIORITY_API_VERSION in api_version and module.params['priority_operator']:
                volume = flasharray.VolumePatch(priority_adjustment=flasharray.PriorityAdjustment(priority_adjustment_operator=module.params['priority_operator'], priority_adjustment_value=module.params['priority_value']))
                prio_res = array.patch_volumes(names=names, volume=volume)
                if prio_res.status_code != 200:
                    array.patch_volumes(names=names, volume=flasharray.VolumePatch(destroyed=True))
                    array.delete_volumes(names=names)
                    module.fail_json(msg='Failed to set DMM Priority Adjustment on volumes. Error: {0}'.format(prio_res.errors[0].message))
                prio_temp = list(prio_res.items)
            temp = list(res.items)
            for count in range(0, len(temp)):
                vol_name = temp[count].name
                volfact[vol_name] = {'size': temp[count].provisioned, 'serial': temp[count].serial, 'created': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(temp[count].created / 1000)), 'page83_naa': PURE_OUI + temp[count].serial.lower(), 'nvme_nguid': _create_nguid(temp[count].serial.lower())}
                if bw_qos_size != 0:
                    volfact[vol_name]['bandwidth_limit'] = temp[count].qos.bandwidth_limit
                if iops_qos_size != 0:
                    volfact[vol_name]['iops_limit'] = temp[count].qos.iops_limit
                if VOLUME_PROMOTION_API_VERSION in api_version and module.params['promotion_state']:
                    volfact[vol_name]['promotion_status'] = prio_temp[count].promotion_status
                if PRIORITY_API_VERSION in api_version and module.params['priority_operator']:
                    volfact[vol_name]['priority_operator'] = prio_temp[count].priority_adjustment.priority_adjustment_operator
                    volfact[vol_name]['priority_value'] = prio_temp[count].priority_adjustment.priority_adjustment_value
    if module.params['pgroup'] and DEFAULT_API_VERSION not in api_version:
        if not module.check_mode:
            res = array.post_protection_groups_volumes(group_names=[module.params['pgroup']], member_names=names)
            if res.status_code != 200:
                module.warn('Failed to add {0} to protection group {1}.'.format(module.params['name'], module.params['pgroup']))
    module.exit_json(changed=changed, volume=volfact)