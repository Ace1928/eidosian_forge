from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_all_dhcpv6_relay_intf_request(self, intf_name):
    """Get request to delete all DHCPv6 relay configurations in the
        specified interface
        """
    return {'path': self.dhcpv6_relay_intf_config_path['server_addresses_all'].format(intf_name=intf_name), 'method': DELETE}