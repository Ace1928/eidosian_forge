from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.static_routes.static_routes import (
def parse_admin_distance(self, item):
    split_item = item.split(' ')
    for item in ['vrf', 'metric', 'tunnel-id', 'vrflabel', 'track', 'tag', 'description']:
        try:
            del split_item[split_item.index(item) + 1]
            del split_item[split_item.index(item)]
        except ValueError:
            continue
    try:
        return [i for i in split_item if '.' not in i and ':' not in i and (ord(i[0]) > 48) and (ord(i[0]) < 57)][0]
    except IndexError:
        return None