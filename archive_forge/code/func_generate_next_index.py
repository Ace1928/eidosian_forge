from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def generate_next_index(vm_filled_indexes_list, num_sign_cnt):
    counter = 0
    cnt_str = str(counter).zfill(num_sign_cnt)
    while cnt_str in vm_filled_indexes_list:
        counter = counter + 1
        cnt_str = str(counter).zfill(num_sign_cnt)
    return cnt_str