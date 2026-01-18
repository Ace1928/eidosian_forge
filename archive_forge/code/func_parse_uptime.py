from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_uptime(self, data):
    match = re.search('^System Up Time \\S+:\\s+(\\S+)\\s*$', data, re.M)
    if match:
        dayhour, mins, sec = match.group(1).split(':')
        day, hour = dayhour.split(',')
        return int(day) * 86400 + int(hour) * 3600 + int(mins) * 60 + int(sec)