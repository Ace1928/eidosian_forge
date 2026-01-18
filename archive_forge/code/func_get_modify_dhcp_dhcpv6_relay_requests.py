from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_modify_dhcp_dhcpv6_relay_requests(self, commands):
    """Get requests to modify DHCP and DHCPv6 relay configurations
        for all interfaces specified by the commands
        """
    requests = []
    for command in commands:
        if command.get('ipv4'):
            requests.extend(self.get_modify_specific_dhcp_relay_param_requests(command))
        if command.get('ipv6'):
            requests.extend(self.get_modify_specific_dhcpv6_relay_param_requests(command))
    return requests