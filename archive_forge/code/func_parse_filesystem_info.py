from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_filesystem_info(self, data):
    match = re.search('Total size of (\\S+): (\\d+) bytes', data, re.M)
    if match:
        self.facts['spacetotal_mb'] = round(int(match.group(2)) / 1024 / 1024, 1)
        match = re.search('Free size of (\\S+): (\\d+) bytes', data, re.M)
        self.facts['spacefree_mb'] = round(int(match.group(2)) / 1024 / 1024, 1)
    else:
        match = re.search('(\\d+)K of (\\d+)K are free', data, re.M)
        if match:
            self.facts['spacetotal_mb'] = round(int(match.group(2)) / 1024, 1)
            self.facts['spacefree_mb'] = round(int(match.group(1)) / 1024, 1)