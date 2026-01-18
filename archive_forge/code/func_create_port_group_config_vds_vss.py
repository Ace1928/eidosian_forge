from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_port_group_config_vds_vss(self):
    port_group_config = vim.host.PortGroup.Config()
    port_group_config.spec = vim.host.PortGroup.Specification()
    port_group_config.changeOperation = 'add'
    port_group_config.spec.name = self.migrate_portgroup_name
    port_group_config.spec.vlanId = self.migrate_vlan_id if self.migrate_vlan_id is not None else 0
    port_group_config.spec.vswitchName = self.migrate_switch_name
    port_group_config.spec.policy = vim.host.NetworkPolicy()
    return port_group_config