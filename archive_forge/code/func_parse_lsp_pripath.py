from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import run_commands
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import ironware_argument_spec, check_args
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_lsp_pripath(self, data):
    match = re.search('Pri\\. path: ([^\\s,]+), up: (\\w+), active: (\\w+)', data, re.M)
    if match:
        path = dict()
        path['name'] = match.group(1) if match.group(1) != 'NONE' else None
        path['up'] = True if match.group(2) == 'yes' else False
        path['active'] = True if match.group(3) == 'yes' else False
        return path