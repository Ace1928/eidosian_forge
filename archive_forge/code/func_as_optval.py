from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
@staticmethod
def as_optval(arg, ignore_none=None):
    return _ArgFormat(lambda value: ['{0}{1}'.format(arg, value)], ignore_none=ignore_none)