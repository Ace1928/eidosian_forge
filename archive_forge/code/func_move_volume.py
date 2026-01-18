from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def move_volume(module, array):
    """Move volume between pods, vgroups or local array"""
    volfact = []
    changed = vgroup_exists = target_exists = pod_exists = False
    api_version = array._list_available_rest_versions()
    pod_name = ''
    vgroup_name = ''
    volume_name = module.params['name']
    if '::' in module.params['name']:
        volume_name = module.params['name'].split('::')[1]
        pod_name = module.params['name'].split('::')[0]
    if '/' in module.params['name']:
        volume_name = module.params['name'].split('/')[1]
        vgroup_name = module.params['name'].split('/')[0]
    if module.params['move'] == 'local':
        target_location = ''
        if '::' not in module.params['name']:
            if '/' not in module.params['name']:
                module.fail_json(msg='Source and destination [local] cannot be the same.')
        try:
            target_exists = array.get_volume(volume_name, pending=True)
        except Exception:
            target_exists = False
        if target_exists:
            module.fail_json(msg='Target volume {0} already exists'.format(volume_name))
    else:
        try:
            pod_exists = array.get_pod(module.params['move'])
            if len(pod_exists['arrays']) > 1:
                module.fail_json(msg='Volume cannot be moved into a stretched pod')
            if pod_exists['link_target_count'] != 0:
                module.fail_json(msg='Volume cannot be moved into a linked source pod')
            if PROMOTE_API_VERSION in api_version:
                if pod_exists['promotion_status'] == 'demoted':
                    module.fail_json(msg='Volume cannot be moved into a demoted pod')
            pod_exists = bool(pod_exists)
        except Exception:
            pod_exists = False
        if pod_exists:
            try:
                target_exists = bool(array.get_volume(module.params['move'] + '::' + volume_name, pending=True))
            except Exception:
                target_exists = False
        try:
            vgroup_exists = bool(array.get_vgroup(module.params['move']))
        except Exception:
            vgroup_exists = False
        if vgroup_exists:
            try:
                target_exists = bool(array.get_volume(module.params['move'] + '/' + volume_name, pending=True))
            except Exception:
                target_exists = False
        if target_exists:
            module.fail_json(msg='Volume of same name already exists in move location')
        if pod_exists and vgroup_exists:
            module.fail_json(msg='Move location {0} matches both a pod and a vgroup. Please rename one of these.'.format(module.params['move']))
        if not pod_exists and (not vgroup_exists):
            module.fail_json(msg='Move location {0} does not exist.'.format(module.params['move']))
        if '::' in module.params['name']:
            pod = array.get_pod(module.params['move'])
            if len(pod['arrays']) > 1:
                module.fail_json(msg='Volume cannot be moved out of a stretched pod')
            if pod['linked_target_count'] != 0:
                module.fail_json(msg='Volume cannot be moved out of a linked source pod')
            if PROMOTE_API_VERSION in api_version:
                if pod['promotion_status'] == 'demoted':
                    module.fail_json(msg='Volume cannot be moved out of a demoted pod')
        if '/' in module.params['name']:
            if vgroup_name == module.params['move'] or pod_name == module.params['move']:
                module.fail_json(msg='Source and destination cannot be the same')
        target_location = module.params['move']
    if get_endpoint(target_location, array):
        module.fail_json(msg='Target volume {0} is a protocol-endpoinnt'.format(target_location))
    changed = True
    if not module.check_mode:
        try:
            volfact = array.move_volume(module.params['name'], target_location)
        except Exception:
            if target_location == '':
                target_location = '[local]'
            module.fail_json(msg='Move of volume {0} to {1} failed.'.format(module.params['name'], target_location))
    module.exit_json(changed=changed, volume=volfact)