from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def rename_volume(module, array):
    """Rename volume within a container, ie pod, vgroup or local array"""
    volfact = []
    changed = False
    pod_name = ''
    vgroup_name = ''
    target_name = module.params['rename']
    target_exists = False
    if '::' in module.params['name']:
        pod_name = module.params['name'].split('::')[0]
        target_name = pod_name + '::' + module.params['rename']
        try:
            array.get_volume(target_name, pending=True)
            target_exists = True
        except Exception:
            target_exists = False
    elif '/' in module.params['name']:
        vgroup_name = module.params['name'].split('/')[0]
        target_name = vgroup_name + '/' + module.params['rename']
        try:
            array.get_volume(target_name, pending=True)
            target_exists = True
        except Exception:
            target_exists = False
    else:
        try:
            array.get_volume(target_name, pending=True)
            target_exists = True
        except Exception:
            target_exists = False
    if target_exists and get_endpoint(target_name, array):
        module.fail_json(msg='Target volume {0} is a protocol-endpoinnt'.format(target_name))
    if not target_exists:
        if get_destroyed_endpoint(target_name, array):
            module.fail_json(msg='Target volume {0} is a destroyed protocol-endpoinnt'.format(target_name))
        else:
            changed = True
            if not module.check_mode:
                try:
                    volfact = array.rename_volume(module.params['name'], module.params['rename'])
                except Exception:
                    module.fail_json(msg='Rename volume {0} to {1} failed.'.format(module.params['name'], module.params['rename']))
    else:
        module.fail_json(msg='Target volume {0} already exists.'.format(target_name))
    module.exit_json(changed=changed, volume=volfact)