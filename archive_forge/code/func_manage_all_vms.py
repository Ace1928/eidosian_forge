from __future__ import absolute_import, division, print_function
import json
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def manage_all_vms(module, vm_state):
    state = module.params['state']
    if state == 'created':
        module.fail_json(msg='State "created" is only valid for tasks with a single VM')
    any_changed = False
    for uuid in get_all_vm_uuids(module):
        current_vm_state = get_vm_prop(module, uuid, 'state')
        if not current_vm_state and vm_state == 'deleted':
            any_changed = False
        elif module.check_mode:
            if not current_vm_state or get_vm_prop(module, uuid, 'state') != state:
                any_changed = True
        else:
            any_changed = vm_state_transition(module, uuid, vm_state) or any_changed
    return any_changed