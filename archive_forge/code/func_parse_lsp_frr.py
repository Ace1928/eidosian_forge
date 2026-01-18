from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import run_commands
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import ironware_argument_spec, check_args
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_lsp_frr(self, data):
    match = re.search('Backup LSP: (\\w+)', data, re.M)
    if match:
        path = dict()
        path['up'] = True if match.group(1) == 'UP' else False
        path['name'] = None
        if path['up']:
            match = re.search('bypass_lsp: (\\S)', data, re.M)
            path['name'] = match.group(1) if match else None
        return path