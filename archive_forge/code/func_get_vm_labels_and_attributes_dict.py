from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def get_vm_labels_and_attributes_dict(client, vm_id):
    vm_USER_TEMPLATE = client.vm.info(vm_id).USER_TEMPLATE
    attrs_dict = {}
    labels_list = []
    for key, value in vm_USER_TEMPLATE.items():
        if key != 'LABELS':
            attrs_dict[key] = value
        elif key is not None and value is not None:
            labels_list = value.split(',')
    return (labels_list, attrs_dict)