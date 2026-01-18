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
def cached_find_executable(exe):
    if exe in exe_cache:
        return exe_cache[exe]
    fc_exe = find_executable(exe)
    exe_cache[exe] = exe_cache[fc_exe] = fc_exe
    return fc_exe