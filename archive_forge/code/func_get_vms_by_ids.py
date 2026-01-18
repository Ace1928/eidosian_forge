from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def get_vms_by_ids(module, client, state, ids):
    vms = []
    for vm_id in ids:
        vm = get_vm_by_id(client, vm_id)
        if vm is None and state != 'absent':
            module.fail_json(msg='There is no VM with id=' + str(vm_id))
        vms.append(vm)
    return vms