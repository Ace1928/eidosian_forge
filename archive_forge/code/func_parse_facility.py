from __future__ import absolute_import, division, print_function
import collections
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import Version
def parse_facility(self, line):
    match = re.search('logging facility (\\S+)', line, re.M)
    facility = None
    if match:
        facility = match.group(1)
    return facility