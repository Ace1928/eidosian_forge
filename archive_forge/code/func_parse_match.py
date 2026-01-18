from __future__ import absolute_import, division, print_function
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def parse_match(match):
    level = None
    if match:
        if int(match.group(1)) in range(0, 8):
            level = match.group(1)
        else:
            pass
    return level