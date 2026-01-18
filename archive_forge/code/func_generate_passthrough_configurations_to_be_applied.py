from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
def generate_passthrough_configurations_to_be_applied(self):
    """
        Generate configs to enable or disable PCI device passthrough.
        The configs are generated against only ESXi host has PCI device to be changed.
        """
    self.host_passthrough_configs = {}
    for esxi_hostname, value in self.host_target_device_to_change_configuration.items():
        self.host_passthrough_configs[esxi_hostname] = {'host_obj': value['host_obj'], 'generated_new_configs': []}
        if value['new_configs']:
            state = True if self.state == 'present' else False
            for new_config in value['new_configs']:
                config = vim.host.PciPassthruConfig()
                config.passthruEnabled = state
                config.id = new_config['device_id']
                self.host_passthrough_configs[esxi_hostname]['generated_new_configs'].append(config)