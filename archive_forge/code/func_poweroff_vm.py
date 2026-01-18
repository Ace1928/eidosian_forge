from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def poweroff_vm(module, client, vm, hard):
    vm = client.vm.info(vm.ID)
    changed = False
    lcm_state = vm.LCM_STATE
    state = vm.STATE
    if lcm_state not in [LCM_STATES.index('SHUTDOWN'), LCM_STATES.index('SHUTDOWN_POWEROFF')] and state not in [VM_STATES.index('POWEROFF')]:
        changed = True
    if changed and (not module.check_mode):
        if not hard:
            client.vm.action('poweroff', vm.ID)
        else:
            client.vm.action('poweroff-hard', vm.ID)
    return changed