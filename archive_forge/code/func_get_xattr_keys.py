from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def get_xattr_keys(module, path, follow):
    cmd = [module.get_bin_path('getfattr', True), '--absolute-names']
    if not follow:
        cmd.append('-h')
    cmd.append(path)
    return _run_xattr(module, cmd)