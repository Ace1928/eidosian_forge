import contextlib
import os
import re
import sys
from distutils.core import Command
from distutils.errors import *
from distutils.sysconfig import customize_compiler, get_python_version
from distutils.sysconfig import get_config_h_filename
from distutils.dep_util import newer_group
from distutils.extension import Extension
from distutils.util import get_platform
from distutils import log
from site import USER_BASE
def find_swig(self):
    """Return the name of the SWIG executable.  On Unix, this is
        just "swig" -- it should be in the PATH.  Tries a bit harder on
        Windows.
        """
    if os.name == 'posix':
        return 'swig'
    elif os.name == 'nt':
        for vers in ('1.3', '1.2', '1.1'):
            fn = os.path.join('c:\\swig%s' % vers, 'swig.exe')
            if os.path.isfile(fn):
                return fn
        else:
            return 'swig.exe'
    else:
        raise DistutilsPlatformError("I don't know how to find (much less run) SWIG on platform '%s'" % os.name)