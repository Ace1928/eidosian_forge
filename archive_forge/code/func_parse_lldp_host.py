from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_lldp_host(self, data):
    match = re.search('System Name: (.+)$', data, re.M)
    if match:
        return match.group(1)