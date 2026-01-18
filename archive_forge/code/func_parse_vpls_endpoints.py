from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import run_commands
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import ironware_argument_spec, check_args
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_vpls_endpoints(self, data):
    facts = list()
    regex = 'Vlan (?P<vlanid>[0-9]+)\\s(?: +(?:L2.*)\\s| +Tagged: (?P<tagged>.+)+\\s| +Untagged: (?P<untagged>.+)\\s)*'
    matches = re.finditer(regex, data, re.IGNORECASE)
    for match in matches:
        f = match.groupdict()
        f['type'] = 'local'
        facts.append(f)
    regex = 'Peer address: (?P<vllpeer>[0-9\\.]+)'
    matches = re.finditer(regex, data, re.IGNORECASE)
    for match in matches:
        f = match.groupdict()
        f['type'] = 'remote'
        facts.append(f)
    return facts