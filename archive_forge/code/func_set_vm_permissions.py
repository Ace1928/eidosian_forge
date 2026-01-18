from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def set_vm_permissions(module, client, vms, permissions):
    changed = False
    for vm in vms:
        vm = client.vm.info(vm.ID)
        old_permissions = parse_vm_permissions(client, vm)
        changed = changed or old_permissions != permissions
        if not module.check_mode and old_permissions != permissions:
            permissions_str = bin(int(permissions, base=8))[2:]
            mode_bits = [int(d) for d in permissions_str]
            try:
                client.vm.chmod(vm.ID, mode_bits[0], mode_bits[1], mode_bits[2], mode_bits[3], mode_bits[4], mode_bits[5], mode_bits[6], mode_bits[7], mode_bits[8])
            except pyone.OneAuthorizationException:
                module.fail_json(msg='Permissions changing is unsuccessful, but instances are present if you deployed them.')
    return changed