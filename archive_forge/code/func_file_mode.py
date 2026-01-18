from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
def file_mode(path):
    if not os.path.exists(path):
        return 0
    return os.stat(path).st_mode & 511