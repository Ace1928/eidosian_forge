from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def get_all_vms_by_attributes(client, attributes_dict, labels_list):
    pool = client.vmpool.info(-2, -1, -1, -1).VM
    vm_list = []
    name = ''
    if attributes_dict:
        name = attributes_dict.pop('NAME', '')
    if name != '':
        base_name = name[:len(name) - name.count('#')]
        with_hash = name.endswith('#')
        for vm in pool:
            if vm.NAME.startswith(base_name):
                if with_hash and vm.NAME[len(base_name):].isdigit():
                    vm_list.append(vm)
                elif not with_hash and vm.NAME == name:
                    vm_list.append(vm)
        pool = vm_list
    import copy
    vm_list = copy.copy(pool)
    for vm in pool:
        remove_list = []
        vm_labels_list, vm_attributes_dict = get_vm_labels_and_attributes_dict(client, vm.ID)
        if attributes_dict and len(attributes_dict) > 0:
            for key, val in attributes_dict.items():
                if key in vm_attributes_dict:
                    if val and vm_attributes_dict[key] != val:
                        remove_list.append(vm)
                        break
                else:
                    remove_list.append(vm)
                    break
        vm_list = list(set(vm_list).difference(set(remove_list)))
        remove_list = []
        if labels_list and len(labels_list) > 0:
            for label in labels_list:
                if label not in vm_labels_list:
                    remove_list.append(vm)
                    break
        vm_list = list(set(vm_list).difference(set(remove_list)))
    return vm_list