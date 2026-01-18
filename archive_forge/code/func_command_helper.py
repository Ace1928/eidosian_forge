from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def command_helper(module, command, errmsg=None):
    """Run a command, catch any nclu errors"""
    _rc, output, _err = module.run_command('/usr/bin/net %s' % command)
    if _rc or 'ERROR' in output or 'ERROR' in _err:
        module.fail_json(msg=errmsg or output)
    return str(output)