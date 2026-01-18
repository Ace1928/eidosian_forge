from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
def normalize_ttl(ttl):
    matches = re.findall('(\\d+)(:h|m|s)', ttl)
    ttl = 0
    for value, unit in matches:
        value = int(value)
        if unit == 'm':
            value *= 60
        elif unit == 'h':
            value *= 60 * 60
        ttl += value
    new_ttl = ''
    hours, remainder = divmod(ttl, 3600)
    if hours:
        new_ttl += '{0}h'.format(hours)
    minutes, seconds = divmod(remainder, 60)
    if minutes:
        new_ttl += '{0}m'.format(minutes)
    if seconds:
        new_ttl += '{0}s'.format(seconds)
    return new_ttl