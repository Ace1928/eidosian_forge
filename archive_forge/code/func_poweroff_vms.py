from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def poweroff_vms(module, client, vms, hard):
    changed = False
    for vm in vms:
        changed = poweroff_vm(module, client, vm, hard) or changed
    return changed