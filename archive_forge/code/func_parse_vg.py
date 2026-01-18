from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def parse_vg(data):
    for line in data.splitlines():
        match = re.search('VOLUME GROUP:\\s+(\\w+)', line)
        if match is not None:
            name = match.group(1)
            continue
        match = re.search('TOTAL PP.*\\((\\d+)', line)
        if match is not None:
            size = int(match.group(1))
            continue
        match = re.search('PP SIZE:\\s+(\\d+)', line)
        if match is not None:
            pp_size = int(match.group(1))
            continue
        match = re.search('FREE PP.*\\((\\d+)', line)
        if match is not None:
            free = int(match.group(1))
            continue
    return {'name': name, 'size': size, 'free': free, 'pp_size': pp_size}