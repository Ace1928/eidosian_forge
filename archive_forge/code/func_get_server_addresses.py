from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
@staticmethod
def get_server_addresses(server_addresses_dict):
    """Get a set of server addresses available in the given
        server_addresses dict
        """
    server_addresses = set()
    if not server_addresses_dict:
        return server_addresses
    for addr in server_addresses_dict:
        if addr.get('address'):
            server_addresses.add(addr['address'])
    return server_addresses