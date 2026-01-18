from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_obj_params(self, get_object):
    """
        Get the required comparison obj_params value

        Parameters:
            get_object (str) - identifier for the required obj_params

        Returns:
            obj_params (list) - obj_params value for comparison.
        """
    try:
        if get_object == 'GlobalPool':
            obj_params = [('settings', 'settings')]
        elif get_object == 'ReservePool':
            obj_params = [('name', 'name'), ('type', 'type'), ('ipv6AddressSpace', 'ipv6AddressSpace'), ('ipv4GlobalPool', 'ipv4GlobalPool'), ('ipv4Prefix', 'ipv4Prefix'), ('ipv4PrefixLength', 'ipv4PrefixLength'), ('ipv4GateWay', 'ipv4GateWay'), ('ipv4DhcpServers', 'ipv4DhcpServers'), ('ipv4DnsServers', 'ipv4DnsServers'), ('ipv6GateWay', 'ipv6GateWay'), ('ipv6DhcpServers', 'ipv6DhcpServers'), ('ipv6DnsServers', 'ipv6DnsServers'), ('ipv4TotalHost', 'ipv4TotalHost'), ('slaacSupport', 'slaacSupport')]
        elif get_object == 'Network':
            obj_params = [('settings', 'settings'), ('site_name', 'site_name')]
        else:
            raise ValueError("Received an unexpected value for 'get_object': {0}".format(get_object))
    except Exception as msg:
        self.log('Received exception: {0}'.format(msg), 'CRITICAL')
    return obj_params