from __future__ import absolute_import, division, print_function
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def parse_facility_link_status(line, facility, status):
    facility_link_status = None
    if facility is not None:
        match = re.search('logging level {0} {1} (\\S+)'.format(facility, status), line, re.M)
        if match:
            facility_link_status = status + '-' + match.group(1)
    return facility_link_status