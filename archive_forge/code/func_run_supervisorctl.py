from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule, is_executable
def run_supervisorctl(cmd, name=None, **kwargs):
    args = list(supervisorctl_args)
    args.append(cmd)
    if name:
        args.append(name)
    return module.run_command(args, **kwargs)