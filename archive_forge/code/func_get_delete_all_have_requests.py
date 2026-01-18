from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_all_have_requests(self, afis):
    """creates and builds list of requests to delete all current dhcp snooping config for ipv4 and ipv6"""
    modified_afi_commands = []
    requests = []
    ipv4_commands, ipv4_requests = self.get_delete_specific_afi_fields_requests(afis.get('have_ipv4'), afis.get('have_ipv4'))
    requests.extend(ipv4_requests)
    if ipv4_commands:
        ipv4_commands['afi'] = afis.get('have_ipv4')['afi']
        modified_afi_commands.append(ipv4_commands)
    ipv6_commands, ipv6_requests = self.get_delete_specific_afi_fields_requests(afis.get('have_ipv6'), afis.get('have_ipv6'))
    requests.extend(ipv6_requests)
    if ipv6_commands:
        ipv6_commands['afi'] = afis.get('have_ipv6')['afi']
        modified_afi_commands.append(ipv6_commands)
    sent_commands = []
    if modified_afi_commands:
        sent_commands = {'afis': modified_afi_commands}
    return (sent_commands, requests)