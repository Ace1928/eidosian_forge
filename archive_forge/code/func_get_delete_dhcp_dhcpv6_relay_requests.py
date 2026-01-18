from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_dhcp_dhcpv6_relay_requests(self, commands, have):
    """Get requests to delete DHCP and DHCPv6 relay configurations
        based on the commands specified
        """
    requests = []
    for command in commands:
        intf_name = command['name']
        have_obj = next((cfg for cfg in have if cfg['name'] == intf_name), None)
        if not have_obj:
            continue
        have_ipv4 = have_obj.get('ipv4')
        have_ipv6 = have_obj.get('ipv6')
        ipv4 = command.get('ipv4')
        ipv6 = command.get('ipv6')
        if not ipv4 and (not ipv6):
            if have_ipv4:
                requests.append(self.get_delete_all_dhcp_relay_intf_request(intf_name))
            if have_ipv6:
                requests.append(self.get_delete_all_dhcpv6_relay_intf_request(intf_name))
        else:
            if ipv4 and have_ipv4:
                requests.extend(self.get_delete_specific_dhcp_relay_param_requests(command, have_obj))
            if ipv6 and have_ipv6:
                requests.extend(self.get_delete_specific_dhcpv6_relay_param_requests(command, have_obj))
    return requests