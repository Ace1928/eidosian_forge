from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def resume_vms(module, client, vms):
    changed = False
    for vm in vms:
        changed = resume_vm(module, client, vm) or changed
    return changed