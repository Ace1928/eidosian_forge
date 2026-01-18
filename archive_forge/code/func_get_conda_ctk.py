import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def get_conda_ctk():
    """Return path to directory containing the shared libraries of cudatoolkit.
    """
    is_conda_env = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    if not is_conda_env:
        return
    paths = find_lib('nvvm')
    if not paths:
        return
    return os.path.dirname(max(paths))