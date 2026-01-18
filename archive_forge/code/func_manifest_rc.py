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
def manifest_rc(name, type='dll'):
    """Return the rc file used to generate the res file which will be embedded
    as manifest for given manifest file name, of given type ('dll' or
    'exe').

    Parameters
    ----------
    name : str
            name of the manifest file to embed
    type : str {'dll', 'exe'}
            type of the binary which will embed the manifest

    """
    if type == 'dll':
        rctype = 2
    elif type == 'exe':
        rctype = 1
    else:
        raise ValueError('Type %s not supported' % type)
    return '#include "winuser.h"\n%d RT_MANIFEST %s' % (rctype, name)