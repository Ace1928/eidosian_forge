from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def set_vm_ownership(module, client, vms, owner_id, group_id):
    changed = False
    for vm in vms:
        vm = client.vm.info(vm.ID)
        if owner_id is None:
            owner_id = vm.UID
        if group_id is None:
            group_id = vm.GID
        changed = changed or owner_id != vm.UID or group_id != vm.GID
        if not module.check_mode and (owner_id != vm.UID or group_id != vm.GID):
            try:
                client.vm.chown(vm.ID, owner_id, group_id)
            except pyone.OneAuthorizationException:
                module.fail_json(msg='Ownership changing is unsuccessful, but instances are present if you deployed them.')
    return changed