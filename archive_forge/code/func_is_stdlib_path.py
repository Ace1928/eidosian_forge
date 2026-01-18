import copy
import sys
import re
import os
from itertools import chain
from contextlib import contextmanager
from parso.python import tree
def is_stdlib_path(path):
    parts = path.parts
    if 'dist-packages' in parts or 'site-packages' in parts:
        return False
    base_path = os.path.join(sys.prefix, 'lib', 'python')
    return bool(re.match(re.escape(base_path) + '\\d.\\d', str(path)))