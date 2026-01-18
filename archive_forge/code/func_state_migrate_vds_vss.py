from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_migrate_vds_vss(self):
    host_network_system = self.host_system.configManager.networkSystem
    config = vim.host.NetworkConfig()
    config.portgroup = [self.create_port_group_config_vds_vss()]
    host_network_system.UpdateNetworkConfig(config, 'modify')
    config = vim.host.NetworkConfig()
    config.vnic = [self.create_host_vnic_config_vds_vss()]
    host_network_system.UpdateNetworkConfig(config, 'modify')
    self.module.exit_json(changed=True)