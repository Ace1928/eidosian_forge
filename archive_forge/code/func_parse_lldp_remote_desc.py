from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
def parse_lldp_remote_desc(self, data):
    match = re.search('Port Description: (.+)$', data, re.M)
    if match:
        return match.group(1)