from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
def generate_system_defaults_config(self):
    system_defaults_config = vim.host.AutoStartManager.SystemDefaults()
    system_defaults_config.enabled = self.system_defaults['enabled']
    system_defaults_config.startDelay = self.system_defaults['start_delay']
    system_defaults_config.stopAction = self.system_defaults['stop_action']
    system_defaults_config.stopDelay = self.system_defaults['stop_delay']
    system_defaults_config.waitForHeartbeat = self.system_defaults['wait_for_heartbeat']
    return system_defaults_config