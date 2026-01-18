from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.static_routes.static_routes import (
def parse_faddr(self, item):
    for x in item.split(' '):
        if (':' in x or '.' in x) and '/' not in x:
            return x