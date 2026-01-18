from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_vol_dict(module, array):
    volume_info = {}
    vols_space = array.list_volumes(space=True)
    vols = array.list_volumes()
    for vol in range(0, len(vols)):
        volume = vols[vol]['name']
        volume_info[volume] = {'protocol_endpoint': False, 'source': vols[vol]['source'], 'size': vols[vol]['size'], 'serial': vols[vol]['serial'], 'page83_naa': PURE_OUI + vols[vol]['serial'], 'nvme_nguid': 'eui.00' + vols[vol]['serial'][0:14].lower() + '24a937' + vols[vol]['serial'][-10:].lower(), 'tags': [], 'hosts': [], 'bandwidth': '', 'iops_limit': '', 'data_reduction': vols_space[vol]['data_reduction'], 'thin_provisioning': vols_space[vol]['thin_provisioning'], 'total_reduction': vols_space[vol]['total_reduction']}
    api_version = array._list_available_rest_versions()
    if V6_MINIMUM_API_VERSION in api_version:
        arrayv6 = get_array(module)
        vols_space = list(arrayv6.get_volumes_space(destroyed=False).items)
        for vol in range(0, len(vols_space)):
            name = vols_space[vol].name
            volume_info[name]['snapshots_space'] = vols_space[vol].space.snapshots
            volume_info[name]['system'] = vols_space[vol].space.unique
            volume_info[name]['unique_space'] = vols_space[vol].space.unique
            volume_info[name]['virtual_space'] = vols_space[vol].space.virtual
            volume_info[name]['total_physical_space'] = vols_space[vol].space.total_physical
            if SHARED_CAP_API_VERSION in api_version:
                volume_info[name]['snapshots_effective'] = getattr(vols_space[vol].space, 'snapshots_effective', None)
                volume_info[name]['unique_effective'] = getattr(vols_space[vol].space, 'unique_effective', None)
                volume_info[name]['total_effective'] = getattr(vols_space[vol].space, 'total_effective', None)
                volume_info[name]['used_provisioned'] = (getattr(vols_space[vol].space, 'used_provisioned', None),)
            if SUBS_API_VERSION in api_version:
                volume_info[name]['total_used'] = vols_space[vol].space.total_used
    if AC_REQUIRED_API_VERSION in api_version:
        qvols = array.list_volumes(qos=True)
        for qvol in range(0, len(qvols)):
            volume = qvols[qvol]['name']
            qos = qvols[qvol]['bandwidth_limit']
            volume_info[volume]['bandwidth'] = qos
            if P53_API_VERSION in api_version:
                iops = qvols[qvol]['iops_limit']
                volume_info[volume]['iops_limit'] = iops
        vvols = array.list_volumes(protocol_endpoint=True)
        for vvol in range(0, len(vvols)):
            volume = vvols[vvol]['name']
            volume_info[volume] = {'protocol_endpoint': True, 'host_encryption_key_status': None, 'source': vvols[vvol]['source'], 'serial': vvols[vvol]['serial'], 'nvme_nguid': 'eui.00' + vvols[vvol]['serial'][0:14].lower() + '24a937' + vvols[vvol]['serial'][-10:].lower(), 'page83_naa': PURE_OUI + vvols[vvol]['serial'], 'tags': [], 'hosts': []}
        if P53_API_VERSION in array._list_available_rest_versions():
            e2ees = array.list_volumes(host_encryption_key=True)
            for e2ee in range(0, len(e2ees)):
                volume = e2ees[e2ee]['name']
                volume_info[volume]['host_encryption_key_status'] = e2ees[e2ee]['host_encryption_key_status']
    if V6_MINIMUM_API_VERSION in api_version:
        volumes = list(arrayv6.get_volumes(destroyed=False).items)
        for vol in range(0, len(volumes)):
            name = volumes[vol].name
            volume_info[name]['promotion_status'] = volumes[vol].promotion_status
            volume_info[name]['requested_promotion_state'] = volumes[vol].requested_promotion_state
            volume_info[name]['subtype'] = volumes[vol].subtype
            if SAFE_MODE_VERSION in api_version:
                volume_info[name]['priority'] = volumes[vol].priority
                volume_info[name]['priority_adjustment'] = volumes[vol].priority_adjustment.priority_adjustment_operator + str(volumes[vol].priority_adjustment.priority_adjustment_value)
    cvols = array.list_volumes(connect=True)
    for cvol in range(0, len(cvols)):
        volume = cvols[cvol]['name']
        voldict = {'host': cvols[cvol]['host'], 'lun': cvols[cvol]['lun']}
        volume_info[volume]['hosts'].append(voldict)
    if ACTIVE_DR_API in api_version:
        voltags = array.list_volumes(tags=True)
        for voltag in range(0, len(voltags)):
            if voltags[voltag]['namespace'] != 'vasa-integration.purestorage.com':
                volume = voltags[voltag]['name']
                tagdict = {'key': voltags[voltag]['key'], 'value': voltags[voltag]['value'], 'copyable': voltags[voltag]['copyable'], 'namespace': voltags[voltag]['namespace']}
                volume_info[volume]['tags'].append(tagdict)
    return volume_info