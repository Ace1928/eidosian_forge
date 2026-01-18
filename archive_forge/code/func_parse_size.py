from __future__ import absolute_import, division, print_function
import collections
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import Version
def parse_size(self, line, dest):
    size = None
    if dest == 'buffered':
        match = re.search('logging buffered (\\S+)', line, re.M)
        if match:
            try:
                int_size = int(match.group(1))
            except ValueError:
                int_size = None
            if int_size is not None:
                if isinstance(int_size, int):
                    size = str(match.group(1))
    if dest == 'file':
        match = re.search('logging file (\\S+) (path\\s\\S+\\s)?maxfilesize (\\S+)', line, re.M)
        if match:
            try:
                if 'path' in line:
                    int_size = int(match.group(2))
                else:
                    int_size = int(match.group(1))
            except ValueError:
                int_size = None
            if int_size is not None:
                if isinstance(int_size, int):
                    size = str(int_size)
    return size