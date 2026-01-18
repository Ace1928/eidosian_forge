from __future__ import absolute_import, division, print_function
import re
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def diversion_state(module, command, path):
    diversion = dict(path=path, state='absent', divert=None, holder=None)
    rc, out, err = module.run_command([command, '--listpackage', path], check_rc=True)
    if out:
        diversion['state'] = 'present'
        diversion['holder'] = out.rstrip()
        rc, out, err = module.run_command([command, '--truename', path], check_rc=True)
        diversion['divert'] = out.rstrip()
    return diversion