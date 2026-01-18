import os
import sys
import stat
import genericpath
from genericpath import *
def splitdrive(p):
    """Split a pathname into drive and path. On Posix, drive is always
    empty."""
    p = os.fspath(p)
    return (p[:0], p)