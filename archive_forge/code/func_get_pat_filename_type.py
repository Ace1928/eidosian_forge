from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, run_commands
from ansible.module_utils.connection import exec_command
def get_pat_filename_type(self, filename):
    """Gets the type of patch filename, such as cfg, zip, dat..."""
    if filename is None:
        return None
    if ' ' in filename:
        self.module.fail_json(msg='Error: Patch file name include spaces.')
    iftype = None
    if filename.endswith('.PAT'):
        iftype = 'PAT'
    else:
        return None
    return iftype.upper()