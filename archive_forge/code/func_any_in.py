from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
def any_in(sequence, *elements):
    return any((e in sequence for e in elements))