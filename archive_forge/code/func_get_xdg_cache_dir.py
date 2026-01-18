import os
import sys
import errno
import shutil
import random
import glob
import warnings
from IPython.utils.process import system
def get_xdg_cache_dir():
    """Return the XDG_CACHE_HOME, if it is defined and exists, else None.

    This is only for non-OS X posix (Linux,Unix,etc.) systems.
    """
    env = os.environ
    if os.name == 'posix':
        xdg = env.get('XDG_CACHE_HOME', None) or os.path.join(get_home_dir(), '.cache')
        if xdg and _writable_dir(xdg):
            assert isinstance(xdg, str)
            return xdg
    return None