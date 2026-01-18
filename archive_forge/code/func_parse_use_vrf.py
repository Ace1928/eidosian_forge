from __future__ import absolute_import, division, print_function
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def parse_use_vrf(line, dest):
    use_vrf = None
    if dest and dest == 'server':
        match = re.search('logging server (?:\\S+) (?:\\d+) use-vrf (\\S+)', line, re.M)
        if match:
            use_vrf = match.group(1)
    return use_vrf