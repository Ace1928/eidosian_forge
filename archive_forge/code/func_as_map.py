from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
@staticmethod
def as_map(_map, default=None, ignore_none=None):
    if default is None:
        default = []
    return _ArgFormat(lambda value: _ensure_list(_map.get(value, default)), ignore_none=ignore_none)