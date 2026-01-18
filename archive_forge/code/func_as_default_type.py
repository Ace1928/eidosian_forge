from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
@staticmethod
def as_default_type(_type, arg='', ignore_none=None):
    fmt = _Format
    if _type == 'dict':
        return fmt.as_func(lambda d: ['--{0}={1}'.format(*a) for a in iteritems(d)], ignore_none=ignore_none)
    if _type == 'list':
        return fmt.as_func(lambda value: ['--{0}'.format(x) for x in value], ignore_none=ignore_none)
    if _type == 'bool':
        return fmt.as_bool('--{0}'.format(arg))
    return fmt.as_opt_val('--{0}'.format(arg), ignore_none=ignore_none)