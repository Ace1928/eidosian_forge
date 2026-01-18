from __future__ import absolute_import, division, print_function
import collections
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import Version
def parse_dest(self, line, group):
    dest_group = ('console', 'monitor', 'buffered', 'file')
    dest = None
    if group in dest_group:
        dest = group
    elif 'vrf' in line:
        dest = 'host'
    return dest