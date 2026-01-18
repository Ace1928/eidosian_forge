import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def get_debian_pkg_libdevice():
    """
    Return the Debian NVIDIA Maintainers-packaged libdevice location, if it
    exists.
    """
    pkg_libdevice_location = '/usr/lib/nvidia-cuda-toolkit/libdevice'
    if not os.path.exists(pkg_libdevice_location):
        return None
    return pkg_libdevice_location