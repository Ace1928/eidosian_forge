import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def get_system_ctk(*subdirs):
    """Return path to system-wide cudatoolkit; or, None if it doesn't exist.
    """
    if sys.platform.startswith('linux'):
        base = '/usr/local/cuda'
        if os.path.exists(base):
            return os.path.join(base, *subdirs)