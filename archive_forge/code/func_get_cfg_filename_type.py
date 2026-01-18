from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, run_commands
from ansible.module_utils.connection import exec_command
def get_cfg_filename_type(self, filename):
    """Gets the type of cfg filename, such as cfg, zip, dat..."""
    if filename is None:
        return None
    if ' ' in filename:
        self.module.fail_json(msg='Error: Configuration file name include spaces.')
    iftype = None
    if filename.endswith('.cfg'):
        iftype = 'cfg'
    elif filename.endswith('.zip'):
        iftype = 'zip'
    elif filename.endswith('.dat'):
        iftype = 'dat'
    else:
        return None
    return iftype.lower()