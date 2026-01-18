from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_specific_requests(self, afis):
    """creates and returns list of requests to delete afi settings.
           Checks if clearing settings for a ip family or just matching fields in config"""
    modified_afi_commands = []
    requests = []
    want_ipv4 = afis.get('want_ipv4')
    want_ipv6 = afis.get('want_ipv6')
    have_ipv4 = afis.get('have_ipv4')
    have_ipv6 = afis.get('have_ipv6')
    if want_ipv4:
        if want_ipv4.keys() == set(['afi']):
            ipv4_commands, ipv4_requests = self.get_delete_specific_afi_fields_requests(have_ipv4, have_ipv4)
        else:
            ipv4_commands, ipv4_requests = self.get_delete_specific_afi_fields_requests(want_ipv4, have_ipv4)
        requests.extend(ipv4_requests)
        if ipv4_commands:
            ipv4_commands['afi'] = want_ipv4['afi']
            modified_afi_commands.append(ipv4_commands)
    if want_ipv6:
        if want_ipv6.keys() == set(['afi']):
            ipv6_commands, ipv6_requests = self.get_delete_specific_afi_fields_requests(have_ipv6, have_ipv6)
        else:
            ipv6_commands, ipv6_requests = self.get_delete_specific_afi_fields_requests(want_ipv6, have_ipv6)
        requests.extend(ipv6_requests)
        if ipv6_commands:
            ipv6_commands['afi'] = want_ipv6['afi']
            modified_afi_commands.append(ipv6_commands)
    sent_commands = []
    if modified_afi_commands:
        sent_commands = {'afis': modified_afi_commands}
    return (sent_commands, requests)