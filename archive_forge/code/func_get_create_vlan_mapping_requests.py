from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_create_vlan_mapping_requests(self, commands, have):
    """ Get list of requests to create/modify vlan mapping configurations
        for all interfaces specified by the commands
        """
    requests = []
    if not commands:
        return requests
    for cmd in commands:
        name = cmd.get('name', None)
        interface_name = name.replace('/', '%2f')
        mapping_list = cmd.get('mapping', [])
        if mapping_list:
            for mapping in mapping_list:
                requests.append(self.get_create_vlan_mapping_request(interface_name, mapping))
    return requests