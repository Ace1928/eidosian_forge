from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.voss.voss import run_commands
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def populate_interfaces_eth_additional(self, interfaces):
    for key, value in iteritems(interfaces):
        match = re.match('^\\w+\\s+\\w+\\s+(\\w+)\\s+(\\d+)\\s+\\w+$', value)
        if match:
            self.facts['interfaces'][key]['description'] = ''
            self.facts['interfaces'][key]['duplex'] = match.group(1)
            self.facts['interfaces'][key]['bandwidth'] = match.group(2)
        else:
            match = re.match('^(.+)\\s+\\w+\\s+\\w+\\s+(\\w+)\\s+(\\d+)\\s+\\w+$', value)
            if match:
                self.facts['interfaces'][key]['description'] = match.group(1).strip()
                self.facts['interfaces'][key]['duplex'] = match.group(2)
                self.facts['interfaces'][key]['bandwidth'] = match.group(3)