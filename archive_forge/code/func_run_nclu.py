from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def run_nclu(module, command_list, command_string, commit, atomic, abort, description):
    _changed = False
    commands = []
    if command_list:
        commands = command_list
    elif command_string:
        commands = command_string.splitlines()
    do_commit = False
    do_abort = abort
    if commit or atomic:
        do_commit = True
        if atomic:
            do_abort = True
    if do_abort:
        command_helper(module, 'abort')
    before = check_pending(module)
    output_lines = []
    for line in commands:
        if line.strip():
            output_lines += [command_helper(module, line.strip(), 'Failed on line %s' % line)]
    output = '\n'.join(output_lines)
    diff = {}
    after = check_pending(module)
    if before == after:
        _changed = False
    else:
        _changed = True
        diff = {'prepared': after}
    if module.check_mode:
        command_helper(module, 'abort')
    if do_commit and _changed and (not module.check_mode):
        result = command_helper(module, "commit description '%s'" % description)
        if 'commit ignored' in result:
            _changed = False
            command_helper(module, 'abort')
        elif command_helper(module, 'show commit last') == '':
            _changed = False
    elif do_abort:
        command_helper(module, 'abort')
    return (_changed, output, diff)