from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def rm_xattr(module, path, key, follow):
    cmd = [module.get_bin_path('setfattr', True)]
    if not follow:
        cmd.append('-h')
    cmd.append('-x')
    cmd.append(key)
    cmd.append(path)
    return _run_xattr(module, cmd, False)