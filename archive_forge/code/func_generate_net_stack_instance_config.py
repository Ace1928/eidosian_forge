from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def generate_net_stack_instance_config(self):
    """
        Generate a new configuration for tcpip stack to modify the configuration.
        """
    self.new_net_stack_instance_configs = vim.host.NetworkConfig()
    self.new_net_stack_instance_configs.netStackSpec = []
    if self.default and self.enabled_net_stack_instance['default']:
        default_config = vim.host.NetworkConfig.NetStackSpec()
        default_config.operation = 'edit'
        default_config.netStackInstance = vim.host.NetStackInstance()
        default_config.netStackInstance.key = self.net_stack_instance_keys['default']
        default_config.netStackInstance.ipRouteConfig = vim.host.IpRouteConfig()
        default_config.netStackInstance.ipRouteConfig.defaultGateway = self.default['gateway']
        default_config.netStackInstance.ipRouteConfig.ipV6DefaultGateway = self.default['ipv6_gateway']
        default_config.netStackInstance.dnsConfig = vim.host.DnsConfig()
        default_config.netStackInstance.dnsConfig.hostName = self.default['hostname']
        default_config.netStackInstance.dnsConfig.domainName = self.default['domain']
        dns_servers = []
        if self.default['preferred_dns']:
            dns_servers.append(self.default['preferred_dns'])
        if self.default['alternate_dns']:
            dns_servers.append(self.default['alternate_dns'])
        default_config.netStackInstance.dnsConfig.address = dns_servers
        default_config.netStackInstance.dnsConfig.searchDomain = self.default['search_domains']
        default_config.netStackInstance.congestionControlAlgorithm = self.default['congestion_algorithm']
        default_config.netStackInstance.requestedMaxNumberOfConnections = self.default['max_num_connections']
        self.new_net_stack_instance_configs.netStackSpec.append(default_config)
    if self.provisioning and self.enabled_net_stack_instance['provisioning']:
        provisioning_config = vim.host.NetworkConfig.NetStackSpec()
        provisioning_config.operation = 'edit'
        provisioning_config.netStackInstance = vim.host.NetStackInstance()
        provisioning_config.netStackInstance.key = self.net_stack_instance_keys['provisioning']
        provisioning_config.netStackInstance.ipRouteConfig = vim.host.IpRouteConfig()
        provisioning_config.netStackInstance.ipRouteConfig.defaultGateway = self.provisioning['gateway']
        provisioning_config.netStackInstance.ipRouteConfig.ipV6DefaultGateway = self.provisioning['ipv6_gateway']
        provisioning_config.netStackInstance.congestionControlAlgorithm = self.provisioning['congestion_algorithm']
        provisioning_config.netStackInstance.requestedMaxNumberOfConnections = self.provisioning['max_num_connections']
        self.new_net_stack_instance_configs.netStackSpec.append(provisioning_config)
    if self.vmotion and self.enabled_net_stack_instance['vmotion']:
        vmotion_config = vim.host.NetworkConfig.NetStackSpec()
        vmotion_config.operation = 'edit'
        vmotion_config.netStackInstance = vim.host.NetStackInstance()
        vmotion_config.netStackInstance.key = self.net_stack_instance_keys['vmotion']
        vmotion_config.netStackInstance.ipRouteConfig = vim.host.IpRouteConfig()
        vmotion_config.netStackInstance.ipRouteConfig.defaultGateway = self.vmotion['gateway']
        vmotion_config.netStackInstance.ipRouteConfig.ipV6DefaultGateway = self.vmotion['ipv6_gateway']
        vmotion_config.netStackInstance.congestionControlAlgorithm = self.vmotion['congestion_algorithm']
        vmotion_config.netStackInstance.requestedMaxNumberOfConnections = self.vmotion['max_num_connections']
        self.new_net_stack_instance_configs.netStackSpec.append(vmotion_config)
    if self.vxlan and self.enabled_net_stack_instance['vxlan']:
        vxlan_config = vim.host.NetworkConfig.NetStackSpec()
        vxlan_config.operation = 'edit'
        vxlan_config.netStackInstance = vim.host.NetStackInstance()
        vxlan_config.netStackInstance.key = self.net_stack_instance_keys['vxlan']
        vxlan_config.netStackInstance.ipRouteConfig = vim.host.IpRouteConfig()
        vxlan_config.netStackInstance.ipRouteConfig.defaultGateway = self.vxlan['gateway']
        vxlan_config.netStackInstance.ipRouteConfig.ipV6DefaultGateway = self.vxlan['ipv6_gateway']
        vxlan_config.netStackInstance.congestionControlAlgorithm = self.vxlan['congestion_algorithm']
        vxlan_config.netStackInstance.requestedMaxNumberOfConnections = self.vxlan['max_num_connections']
        self.new_net_stack_instance_configs.netStackSpec.append(vxlan_config)