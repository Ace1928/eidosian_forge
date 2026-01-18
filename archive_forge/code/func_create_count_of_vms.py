from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def create_count_of_vms(module, client, template_id, count, attributes_dict, labels_list, disk_size, network_attrs_list, wait, wait_timeout, vm_start_on_hold, vm_persistent, updateconf_dict):
    new_vms_list = []
    vm_name = ''
    if attributes_dict:
        vm_name = attributes_dict.get('NAME', '')
    if module.check_mode:
        return (True, [], [])
    vm_filled_indexes_list = None
    num_sign_cnt = vm_name.count('#')
    if vm_name != '' and num_sign_cnt > 0:
        vm_list = get_all_vms_by_attributes(client, {'NAME': vm_name}, None)
        base_name = vm_name[:len(vm_name) - num_sign_cnt]
        vm_name = base_name
        vm_filled_indexes_list = list((vm.NAME[len(base_name):].zfill(num_sign_cnt) for vm in vm_list))
    while count > 0:
        new_vm_name = vm_name
        if vm_filled_indexes_list is not None:
            next_index = generate_next_index(vm_filled_indexes_list, num_sign_cnt)
            vm_filled_indexes_list.append(next_index)
            new_vm_name += next_index
        attributes_dict['NAME'] = new_vm_name
        new_vm_dict = create_vm(module, client, template_id, attributes_dict, labels_list, disk_size, network_attrs_list, vm_start_on_hold, vm_persistent, updateconf_dict)
        new_vm_id = new_vm_dict.get('vm_id')
        new_vm = get_vm_by_id(client, new_vm_id)
        new_vms_list.append(new_vm)
        count -= 1
    if vm_start_on_hold:
        if wait:
            for vm in new_vms_list:
                wait_for_hold(module, client, vm, wait_timeout)
    elif wait:
        for vm in new_vms_list:
            wait_for_running(module, client, vm, wait_timeout)
    return (True, new_vms_list, [])