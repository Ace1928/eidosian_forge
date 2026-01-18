from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_ipv4(self, data):
    match = re.search('Internet address is (\\S+)', data)
    if match:
        addr, masklen = match.group(1).split('/')
        return dict(address=addr, masklen=int(masklen))