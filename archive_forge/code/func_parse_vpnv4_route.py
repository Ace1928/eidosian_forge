from __future__ import (absolute_import, division, print_function)
import re
from ansible_collections.community.routeros.plugins.module_utils.routeros import run_commands
from ansible_collections.community.routeros.plugins.module_utils.routeros import routeros_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_vpnv4_route(self, data):
    facts = dict()
    data = self.preprocess(data)
    for line in data:
        name = self.parse_interface(line)
        facts[name] = dict()
        for key, value in re.findall(self.DETAIL_RE, line):
            facts[name][key] = value
    return facts