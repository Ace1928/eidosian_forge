from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule, is_executable
def take_action_on_processes(processes, status_filter, action, expected_result, exit_module=True):
    to_take_action_on = []
    for process_name, status in processes:
        if status_filter(status):
            to_take_action_on.append(process_name)
    if len(to_take_action_on) == 0:
        if not exit_module:
            return
        module.exit_json(changed=False, name=name, state=state)
    if module.check_mode:
        if not exit_module:
            return
        module.exit_json(changed=True)
    for process_name in to_take_action_on:
        rc, out, err = run_supervisorctl(action, process_name, check_rc=True)
        if '%s: %s' % (process_name, expected_result) not in out:
            module.fail_json(msg=out)
    if exit_module:
        module.exit_json(changed=True, name=name, state=state, affected=to_take_action_on)