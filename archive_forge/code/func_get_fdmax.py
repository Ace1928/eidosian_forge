import errno
import numbers
import os
import subprocess
import sys
from itertools import zip_longest
from io import UnsupportedOperation
def get_fdmax(default=None):
    """Return the maximum number of open file descriptors
    on this system.

    :keyword default: Value returned if there's no file
                      descriptor limit.

    """
    try:
        return os.sysconf('SC_OPEN_MAX')
    except:
        pass
    if resource is None:
        return default
    fdmax = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if fdmax == resource.RLIM_INFINITY:
        return default
    return fdmax