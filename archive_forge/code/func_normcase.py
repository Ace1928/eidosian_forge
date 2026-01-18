import os
import sys
import stat
import genericpath
from genericpath import *
def normcase(s):
    """Normalize case of pathname.  Has no effect under Posix"""
    return os.fspath(s)