from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import math
import re
import os
def parse_unit(size_str, unit=''):
    """
    Parses a string containing a size or boundary information
    """
    matches = re.search('^(-?[\\d.]+) *([\\w%]+)?$', size_str)
    if matches is None:
        matches = re.search('^(\\d+),(\\d+),(\\d+)$', size_str)
        if matches is None:
            module.fail_json(msg="Error interpreting parted size output: '%s'" % size_str)
        size = {'cylinder': int(matches.group(1)), 'head': int(matches.group(2)), 'sector': int(matches.group(3))}
        unit = 'chs'
    else:
        if matches.group(2) is not None:
            unit = matches.group(2)
        size = float(matches.group(1))
    return (size, unit)