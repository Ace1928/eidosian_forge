import os
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler  # noqa: F401
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import (msvc_runtime_library,
def build_import_library():
    if os.name != 'nt':
        return
    arch = get_build_architecture()
    if arch == 'AMD64':
        return _build_import_library_amd64()
    elif arch == 'Intel':
        return _build_import_library_x86()
    else:
        raise ValueError('Unhandled arch %s' % arch)