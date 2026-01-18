import enum
import os
import platform
import sys
import cffi
def set_keepcaps(enable):
    """Set/unset thread's "keep capabilities" flag - see prctl(2)"""
    ret = _prctl(crt.PR_SET_KEEPCAPS, ffi.cast('unsigned long', bool(enable)))
    if ret != 0:
        errno = ffi.errno
        raise OSError(errno, os.strerror(errno))