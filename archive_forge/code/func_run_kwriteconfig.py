from __future__ import (absolute_import, division, print_function)
import os
import shutil
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
def run_kwriteconfig(module, cmd, path, groups, key, value):
    """Invoke kwriteconfig with arguments"""
    args = [cmd, '--file', path, '--key', key]
    for group in groups:
        args.extend(['--group', group])
    if isinstance(value, bool):
        args.extend(['--type', 'bool'])
        if value:
            args.append('true')
        else:
            args.append('false')
    else:
        args.append(value)
    module.run_command(args, check_rc=True)