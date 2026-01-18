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
def find_executables(self):
    """Go through the self.executables dictionary, and attempt to
        find and assign appropriate executables.

        Executable names are looked for in the environment (environment
        variables, the distutils.cfg, and command line), the 0th-element of
        the command list, and the self.possible_executables list.

        Also, if the 0th element is "<F77>" or "<F90>", the Fortran 77
        or the Fortran 90 compiler executable is used, unless overridden
        by an environment setting.

        Subclasses should call this if overridden.
        """
    assert self._is_customised
    exe_cache = self._exe_cache

    def cached_find_executable(exe):
        if exe in exe_cache:
            return exe_cache[exe]
        fc_exe = find_executable(exe)
        exe_cache[exe] = exe_cache[fc_exe] = fc_exe
        return fc_exe

    def verify_command_form(name, value):
        if value is not None and (not is_sequence_of_strings(value)):
            raise ValueError('%s value %r is invalid in class %s' % (name, value, self.__class__.__name__))

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
    ctype = self.compiler_type
    f90 = set_exe('compiler_f90')
    if not f90:
        f77 = set_exe('compiler_f77')
        if f77:
            log.warn('%s: no Fortran 90 compiler found' % ctype)
        else:
            raise CompilerNotFound('%s: f90 nor f77' % ctype)
    else:
        f77 = set_exe('compiler_f77', f90=f90)
        if not f77:
            log.warn('%s: no Fortran 77 compiler found' % ctype)
        set_exe('compiler_fix', f90=f90)
    set_exe('linker_so', f77=f77, f90=f90)
    set_exe('linker_exe', f77=f77, f90=f90)
    set_exe('version_cmd', f77=f77, f90=f90)
    set_exe('archiver')
    set_exe('ranlib')