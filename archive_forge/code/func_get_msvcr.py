import os
import sys
import copy
from subprocess import Popen, PIPE, check_output
import re
from distutils.unixccompiler import UnixCCompiler
from distutils.file_util import write_file
from distutils.errors import (DistutilsExecError, CCompilerError,
from distutils.version import LooseVersion
from distutils.spawn import find_executable
def get_msvcr():
    """Include the appropriate MSVC runtime library if Python was built
    with MSVC 7.0 or later.
    """
    msc_pos = sys.version.find('MSC v.')
    if msc_pos != -1:
        msc_ver = sys.version[msc_pos + 6:msc_pos + 10]
        if msc_ver == '1300':
            return ['msvcr70']
        elif msc_ver == '1310':
            return ['msvcr71']
        elif msc_ver == '1400':
            return ['msvcr80']
        elif msc_ver == '1500':
            return ['msvcr90']
        elif msc_ver == '1600':
            return ['msvcr100']
        else:
            raise ValueError('Unknown MS Compiler version %s ' % msc_ver)