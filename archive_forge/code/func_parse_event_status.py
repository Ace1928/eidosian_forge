from __future__ import absolute_import, division, print_function
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def parse_event_status(line, event):
    status = None
    match = re.search('logging event {0} (\\S+)'.format(event + '-status'), line, re.M)
    if match:
        state = match.group(1)
        if state:
            status = state
    return status