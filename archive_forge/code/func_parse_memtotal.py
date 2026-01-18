from __future__ import absolute_import, division, print_function
import re
import json
from ansible_collections.community.network.plugins.module_utils.network.exos.exos import run_commands
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_memtotal(self, data):
    match = re.search(' Total DRAM \\(KB\\): (\\d+)', data, re.M)
    if match:
        return match.group(1)
    match = re.search(' Total \\s+\\(KB\\): (\\d+)', data, re.M)
    if match:
        return match.group(1)