from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def parse_vm_permissions(client, vm):
    vm_PERMISSIONS = client.vm.info(vm.ID).PERMISSIONS
    owner_octal = int(vm_PERMISSIONS.OWNER_U) * 4 + int(vm_PERMISSIONS.OWNER_M) * 2 + int(vm_PERMISSIONS.OWNER_A)
    group_octal = int(vm_PERMISSIONS.GROUP_U) * 4 + int(vm_PERMISSIONS.GROUP_M) * 2 + int(vm_PERMISSIONS.GROUP_A)
    other_octal = int(vm_PERMISSIONS.OTHER_U) * 4 + int(vm_PERMISSIONS.OTHER_M) * 2 + int(vm_PERMISSIONS.OTHER_A)
    permissions = str(owner_octal) + str(group_octal) + str(other_octal)
    return permissions