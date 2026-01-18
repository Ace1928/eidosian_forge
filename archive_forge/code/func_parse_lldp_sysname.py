from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def parse_lldp_sysname(self, data):
    match = re.search('Device ID:(.+)$', data, re.M)
    if match:
        return match.group(1)