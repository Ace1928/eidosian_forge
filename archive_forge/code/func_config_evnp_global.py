from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def config_evnp_global(self):
    """ set global EVPN configuration"""
    if not self.conf_exist:
        if self.overlay_enable == 'enable':
            self.cli_add_command('evpn-overlay enable')
        else:
            self.cli_add_command('evpn-overlay enable', True)
        if self.commands:
            self.cli_load_config(self.commands)
            self.changed = True