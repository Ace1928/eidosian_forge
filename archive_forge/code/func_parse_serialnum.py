from __future__ import absolute_import, division, print_function
import platform
import re
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def parse_serialnum(self, data):
    match = re.search('(?:HW|Hardware) S/N:\\s+(\\S+)', data)
    if match:
        return match.group(1)