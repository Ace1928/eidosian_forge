from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_server(line, dest):
    if dest == 'server':
        vrf, server = (None, None)
        match = re.search('(ntp\\sserver\\s)(vrf\\s\\w+\\s)?(\\d+\\.\\d+\\.\\d+\\.\\d+)', line, re.M)
        if match and match.group(2) and match.group(3):
            vrf = match.group(2)
            server = match.group(3)
            return (vrf, server)
        if match and match.group(3):
            server = match.group(3)
            return (vrf, server)