from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def terminate_vm(module, client, vm, hard=False):
    changed = False
    if not vm:
        return changed
    changed = True
    if not module.check_mode:
        if hard:
            client.vm.action('terminate-hard', vm.ID)
        else:
            client.vm.action('terminate', vm.ID)
    return changed