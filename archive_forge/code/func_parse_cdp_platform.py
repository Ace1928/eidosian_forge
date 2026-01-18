from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_cdp_platform(self, data):
    match = re.search('^Platform: (.+),', data, re.M)
    if match:
        return match.group(1)