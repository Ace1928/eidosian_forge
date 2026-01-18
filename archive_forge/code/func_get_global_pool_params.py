from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_global_pool_params(self, pool_info):
    """
        Process Global Pool params from playbook data for Global Pool config in Cisco Catalyst Center

        Parameters:
            pool_info (dict) - Playbook data containing information about the global pool

        Returns:
            dict or None - Processed Global Pool data in a format suitable
            for Cisco Catalyst Center configuration, or None if pool_info is empty.
        """
    if not pool_info:
        self.log('Global Pool is empty', 'INFO')
        return None
    self.log('Global Pool Details: {0}'.format(pool_info), 'DEBUG')
    global_pool = {'settings': {'ippool': [{'dhcpServerIps': pool_info.get('dhcpServerIps'), 'dnsServerIps': pool_info.get('dnsServerIps'), 'ipPoolCidr': pool_info.get('ipPoolCidr'), 'ipPoolName': pool_info.get('ipPoolName'), 'type': pool_info.get('ipPoolType').capitalize()}]}}
    self.log('Formated global pool details: {0}'.format(global_pool), 'DEBUG')
    global_ippool = global_pool.get('settings').get('ippool')[0]
    if pool_info.get('ipv6') is False:
        global_ippool.update({'IpAddressSpace': 'IPv4'})
    else:
        global_ippool.update({'IpAddressSpace': 'IPv6'})
    self.log('ip_address_space: {0}'.format(global_ippool.get('IpAddressSpace')), 'DEBUG')
    if not pool_info['gateways']:
        global_ippool.update({'gateway': ''})
    else:
        global_ippool.update({'gateway': pool_info.get('gateways')[0]})
    return global_pool