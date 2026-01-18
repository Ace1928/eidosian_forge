import os
import sys
import re
from pathlib import Path
from distutils.sysconfig import get_python_lib
from distutils.fancy_getopt import FancyGetopt
from distutils.errors import DistutilsModuleError, \
from distutils.util import split_quoted, strtobool
from numpy.distutils.ccompiler import CCompiler, gen_lib_options
from numpy.distutils import log
from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
from numpy.distutils.exec_command import find_executable
from numpy.distutils import _shell_utils
from .environment import EnvironmentConfig
def available_fcompilers_for_platform(osname=None, platform=None):
    if osname is None:
        osname = os.name
    if platform is None:
        platform = sys.platform
    matching_compiler_types = []
    for pattern, compiler_type in _default_compilers:
        if re.match(pattern, platform) or re.match(pattern, osname):
            for ct in compiler_type:
                if ct not in matching_compiler_types:
                    matching_compiler_types.append(ct)
    if not matching_compiler_types:
        matching_compiler_types.append('gnu')
    return matching_compiler_types