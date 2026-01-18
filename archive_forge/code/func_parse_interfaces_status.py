from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import run_commands
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_interfaces_status(self, data, interfaces):
    for line in data.split('\n'):
        match = re.match('(\\d\\/\\d+)', line)
        if match:
            name = match.group(1)
            interface = interfaces[name]
            interface['physicalstatus'] = line[61:71].strip()
            interface['mediatype'] = line[73:91].strip()