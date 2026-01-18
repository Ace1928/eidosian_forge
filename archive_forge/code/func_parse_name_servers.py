from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_name_servers(config):
    match = re.findall('^ip name-server (?:vrf (\\S+) )*(.*)', config, re.M)
    matches = list()
    for vrf, servers in match:
        if not vrf:
            vrf = None
        for server in servers.split():
            matches.append({'server': server, 'vrf': vrf})
    return matches