from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_cpu_load(self, data):
    match = re.search('one minute:\\s+(\\d+)%;\\s*', data, re.M)
    if match:
        return match.group(1)