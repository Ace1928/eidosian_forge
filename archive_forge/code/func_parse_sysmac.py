from __future__ import absolute_import, division, print_function
import re
import json
from ansible_collections.community.network.plugins.module_utils.network.exos.exos import run_commands
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_sysmac(self, data):
    match = re.search('System MAC:\\s+(\\S+)', data, re.M)
    if match:
        return match.group(1)