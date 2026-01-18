from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_want_reserve_pool(self, reserve_pool):
    """
        Get all the Reserved Pool information from playbook
        Set the status and the msg before returning from the API
        Check the return value of the API with check_return_status()

        Parameters:
            reserve_pool (dict) - Playbook reserved pool
            details containing various properties.

        Returns:
            self - The current object with updated desired Reserved Pool information.
        """
    want_reserve = {'name': reserve_pool.get('name'), 'type': reserve_pool.get('pool_type'), 'ipv6AddressSpace': reserve_pool.get('ipv6_address_space'), 'ipv4GlobalPool': reserve_pool.get('ipv4_global_pool'), 'ipv4Prefix': reserve_pool.get('ipv4_prefix'), 'ipv4PrefixLength': reserve_pool.get('ipv4_prefix_length'), 'ipv4GateWay': reserve_pool.get('ipv4_gateway'), 'ipv4DhcpServers': reserve_pool.get('ipv4_dhcp_servers'), 'ipv4DnsServers': reserve_pool.get('ipv4_dns_servers'), 'ipv4Subnet': reserve_pool.get('ipv4_subnet'), 'ipv6GlobalPool': reserve_pool.get('ipv6_global_pool'), 'ipv6Prefix': reserve_pool.get('ipv6_prefix'), 'ipv6PrefixLength': reserve_pool.get('ipv6_prefix_length'), 'ipv6GateWay': reserve_pool.get('ipv6_gateway'), 'ipv6DhcpServers': reserve_pool.get('ipv6_dhcp_servers'), 'ipv6Subnet': reserve_pool.get('ipv6_subnet'), 'ipv6DnsServers': reserve_pool.get('ipv6_dns_servers'), 'ipv4TotalHost': reserve_pool.get('ipv4_total_host'), 'ipv6TotalHost': reserve_pool.get('ipv6_total_host')}
    if not want_reserve.get('name'):
        self.msg = "Missing mandatory parameter 'name' in reserve_pool_details"
        self.status = 'failed'
        return self
    if want_reserve.get('ipv4Prefix') is True:
        if want_reserve.get('ipv4Subnet') is None and want_reserve.get('ipv4TotalHost') is None:
            self.msg = "missing parameter 'ipv4_subnet' or 'ipv4TotalHost'                     while adding the ipv4 in reserve_pool_details"
            self.status = 'failed'
            return self
    if want_reserve.get('ipv6Prefix') is True:
        if want_reserve.get('ipv6Subnet') is None and want_reserve.get('ipv6TotalHost') is None:
            self.msg = "missing parameter 'ipv6_subnet' or 'ipv6TotalHost'                     while adding the ipv6 in reserve_pool_details"
            self.status = 'failed'
            return self
    self.log('Reserved IP pool playbook details: {0}'.format(want_reserve), 'DEBUG')
    if not self.have.get('reservePool').get('details'):
        if not want_reserve.get('ipv4GlobalPool'):
            self.msg = "missing parameter 'ipv4GlobalPool' in reserve_pool_details"
            self.status = 'failed'
            return self
        if not want_reserve.get('ipv4PrefixLength'):
            self.msg = "missing parameter 'ipv4_prefix_length' in reserve_pool_details"
            self.status = 'failed'
            return self
        if want_reserve.get('type') is None:
            want_reserve.update({'type': 'Generic'})
        if want_reserve.get('ipv4GateWay') is None:
            want_reserve.update({'ipv4GateWay': ''})
        if want_reserve.get('ipv4DhcpServers') is None:
            want_reserve.update({'ipv4DhcpServers': []})
        if want_reserve.get('ipv4DnsServers') is None:
            want_reserve.update({'ipv4DnsServers': []})
        if want_reserve.get('ipv6AddressSpace') is None:
            want_reserve.update({'ipv6AddressSpace': False})
        if want_reserve.get('slaacSupport') is None:
            want_reserve.update({'slaacSupport': True})
        if want_reserve.get('ipv4TotalHost') is None:
            del want_reserve['ipv4TotalHost']
        if want_reserve.get('ipv6AddressSpace') is True:
            want_reserve.update({'ipv6Prefix': True})
        else:
            del want_reserve['ipv6Prefix']
        if not want_reserve.get('ipv6AddressSpace'):
            keys_to_check = ['ipv6GlobalPool', 'ipv6PrefixLength', 'ipv6GateWay', 'ipv6DhcpServers', 'ipv6DnsServers', 'ipv6TotalHost']
            for key in keys_to_check:
                if want_reserve.get(key) is None:
                    del want_reserve[key]
    else:
        keys_to_delete = ['type', 'ipv4GlobalPool', 'ipv4Prefix', 'ipv4PrefixLength', 'ipv4TotalHost', 'ipv4Subnet']
        for key in keys_to_delete:
            if key in want_reserve:
                del want_reserve[key]
    self.want.update({'wantReserve': want_reserve})
    self.log('Desired State (want): {0}'.format(self.want), 'INFO')
    self.msg = 'Collecting the reserve pool details from the playbook'
    self.status = 'success'
    return self