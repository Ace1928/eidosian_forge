import os
import sys
import stat
import genericpath
from genericpath import *
def isabs(s):
    """Test whether a path is absolute"""
    s = os.fspath(s)
    sep = _get_sep(s)
    return s.startswith(sep)