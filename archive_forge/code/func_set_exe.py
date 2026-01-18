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
def set_exe(exe_key, f77=None, f90=None):
    cmd = self.executables.get(exe_key, None)
    if not cmd:
        return None
    exe_from_environ = getattr(self.command_vars, exe_key)
    if not exe_from_environ:
        possibles = [f90, f77] + self.possible_executables
    else:
        possibles = [exe_from_environ] + self.possible_executables
    seen = set()
    unique_possibles = []
    for e in possibles:
        if e == '<F77>':
            e = f77
        elif e == '<F90>':
            e = f90
        if not e or e in seen:
            continue
        seen.add(e)
        unique_possibles.append(e)
    for exe in unique_possibles:
        fc_exe = cached_find_executable(exe)
        if fc_exe:
            cmd[0] = fc_exe
            return fc_exe
    self.set_command(exe_key, None)
    return None