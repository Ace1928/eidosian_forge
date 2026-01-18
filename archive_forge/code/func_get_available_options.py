from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def get_available_options(module, command='install'):
    rc, out, err = composer_command(module, 'help %s' % command, arguments='--no-interaction --format=json')
    if rc != 0:
        output = parse_out(err)
        module.fail_json(msg=output)
    command_help_json = module.from_json(out)
    return command_help_json['definition']['options']