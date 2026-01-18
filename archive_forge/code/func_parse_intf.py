from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.static_routes.static_routes import (
def parse_intf(self, item):
    inf_search_strs = [' ((\\w+)((?:\\d)/(?:\\d)/(?:\\d)/(?:\\d+)))', ' (([a-zA-Z]+)(?:\\d+))']
    for i in inf_search_strs:
        match = re.search(i, item, re.M)
        if match:
            return match.group(1)