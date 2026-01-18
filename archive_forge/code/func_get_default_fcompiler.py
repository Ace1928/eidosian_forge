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
def get_default_fcompiler(osname=None, platform=None, requiref90=False, c_compiler=None):
    """Determine the default Fortran compiler to use for the given
    platform."""
    matching_compiler_types = available_fcompilers_for_platform(osname, platform)
    log.info("get_default_fcompiler: matching types: '%s'", matching_compiler_types)
    compiler_type = _find_existing_fcompiler(matching_compiler_types, osname=osname, platform=platform, requiref90=requiref90, c_compiler=c_compiler)
    return compiler_type