from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_boot_version(self, data):
    match = re.search('Boot version\\s*(\\S+)\\s*.*$', data, re.M)
    if match:
        return match.group(1)