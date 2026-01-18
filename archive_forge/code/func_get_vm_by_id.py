from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def get_vm_by_id(client, vm_id):
    try:
        vm = client.vm.info(int(vm_id))
    except BaseException:
        return None
    return vm