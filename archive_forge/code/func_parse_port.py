from __future__ import absolute_import, division, print_function
import platform
import re
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def parse_port(self, data):
    match = re.search('PortDescr:\\s+(.+)$', data, re.M)
    if match:
        return match.group(1)