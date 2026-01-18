from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
def cli_load_config(self, commands):
    """ Cli load configuration """
    if not self.module.check_mode:
        load_config(self.module, commands)