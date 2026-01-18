from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def parse_lvs(data):
    lvs = []
    for line in data.splitlines():
        parts = line.strip().split(';')
        lvs.append({'name': parts[0].replace('[', '').replace(']', ''), 'size': float(parts[1]), 'active': parts[2][4] == 'a', 'thinpool': parts[2][0] == 't', 'thinvol': parts[2][0] == 'V'})
    return lvs