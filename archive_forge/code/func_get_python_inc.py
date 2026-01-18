import _imp
import os
import re
import sys
import warnings
from functools import partial
from .errors import DistutilsPlatformError
from sysconfig import (
def get_python_inc(plat_specific=0, prefix=None):
    """Return the directory containing installed Python header files.

    If 'plat_specific' is false (the default), this is the path to the
    non-platform-specific header files, i.e. Python.h and so on;
    otherwise, this is the path to platform-specific header files
    (namely pyconfig.h).

    If 'prefix' is supplied, use it instead of sys.base_prefix or
    sys.base_exec_prefix -- i.e., ignore 'plat_specific'.
    """
    if prefix is None:
        prefix = plat_specific and BASE_EXEC_PREFIX or BASE_PREFIX
    if os.name == 'posix':
        if python_build:
            if plat_specific:
                return project_base
            else:
                incdir = os.path.join(get_config_var('srcdir'), 'Include')
                return os.path.normpath(incdir)
        python_dir = 'python' + get_python_version() + build_flags
        return os.path.join(prefix, 'include', python_dir)
    elif os.name == 'nt':
        if python_build:
            return os.path.join(prefix, 'include') + os.path.pathsep + os.path.join(prefix, 'PC')
        return os.path.join(prefix, 'include')
    else:
        raise DistutilsPlatformError("I don't know where Python installs its C header files on platform '%s'" % os.name)