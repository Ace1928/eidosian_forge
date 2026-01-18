from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
def mpint(self, value):
    if not isinstance(value, (int, long)):
        raise TypeError('Value must be of type (long, int) not %s' % type(value))
    self.string(self._int_to_mpint(value))
    return self