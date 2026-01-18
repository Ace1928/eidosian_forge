from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import math
import re
import os
def check_size_format(size_str):
    """
    Checks if the input string is an allowed size
    """
    size, unit = parse_unit(size_str)
    return unit in parted_units