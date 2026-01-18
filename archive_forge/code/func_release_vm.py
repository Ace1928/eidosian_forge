from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def release_vm(module, client, vm):
    vm = client.vm.info(vm.ID)
    changed = False
    state = vm.STATE
    if state != VM_STATES.index('HOLD'):
        module.fail_json(msg="Cannot perform action 'release' because this action is not available " + "because VM is not in state 'HOLD'.")
    else:
        changed = True
    if changed and (not module.check_mode):
        client.vm.action('release', vm.ID)
    return changed