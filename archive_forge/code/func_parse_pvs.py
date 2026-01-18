from __future__ import absolute_import, division, print_function
import itertools
import os
from ansible.module_utils.basic import AnsibleModule
def parse_pvs(module, data):
    pvs = []
    dm_prefix = '/dev/dm-'
    for line in data.splitlines():
        parts = line.strip().split(';')
        if parts[0].startswith(dm_prefix):
            parts[0] = find_mapper_device_name(module, parts[0])
        pvs.append({'name': parts[0], 'vg_name': parts[1]})
    return pvs