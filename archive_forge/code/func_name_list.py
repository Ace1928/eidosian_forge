from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
def name_list(self, value):
    if not isinstance(value, list):
        raise TypeError('Value must be a list of byte strings not %s' % type(value))
    try:
        self.string(','.join(value).encode('ASCII'))
    except UnicodeEncodeError as e:
        raise ValueError("Name-list's must consist of US-ASCII characters: %s" % e)
    return self