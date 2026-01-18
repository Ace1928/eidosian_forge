from __future__ import absolute_import, division, print_function
import re
import os
import math
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def size_string(value):
    """Convert a raw value to a string, but only if it is an integer, a float
       or a string itself.
    """
    if not isinstance(value, (int, float, str)):
        raise AssertionError('invalid value type (%s): size must be integer, float or string' % type(value))
    return str(value)