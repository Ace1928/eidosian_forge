from __future__ import absolute_import, division, print_function
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def parse_facility_level(line, facility, dest):
    facility_level = None
    if dest == 'server':
        match = re.search('logging server (?:\\S+) (\\d+)', line, re.M)
        if match:
            facility_level = match.group(1)
    elif facility is not None:
        match = re.search('logging level {0} (\\S+)'.format(facility), line, re.M)
        if match:
            facility_level = match.group(1)
    return facility_level