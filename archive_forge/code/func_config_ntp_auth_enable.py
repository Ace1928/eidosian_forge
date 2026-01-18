from __future__ import (absolute_import, division, print_function)
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
def config_ntp_auth_enable(self):
    """Config ntp authentication enable"""
    commands = list()
    if self.ntp_auth_conf['authentication'] != self.authentication:
        if self.authentication == 'enable':
            config_cli = 'ntp authentication enable'
        else:
            config_cli = 'undo ntp authentication enable'
        commands.append(config_cli)
        self.cli_load_config(commands)