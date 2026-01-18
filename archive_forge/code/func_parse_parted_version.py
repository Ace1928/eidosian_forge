from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import math
import re
import os
def parse_parted_version(out):
    """
    Returns version tuple from the output of "parted --version" command
    """
    lines = [x for x in out.split('\n') if x.strip() != '']
    if len(lines) == 0:
        return (None, None, None)
    matches = re.search('^parted.+\\s(\\d+)\\.(\\d+)(?:\\.(\\d+))?', lines[0].strip())
    if matches is None:
        return (None, None, None)
    major = int(matches.group(1))
    minor = int(matches.group(2))
    rev = 0
    if matches.group(3) is not None:
        rev = int(matches.group(3))
    return (major, minor, rev)