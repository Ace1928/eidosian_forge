from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def config_netstream_export(self):
    """configure netstream export"""
    if self.commands:
        self.cli_load_config(self.commands)
        self.changed = True