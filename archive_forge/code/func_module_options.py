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
def module_options(self, module_dirs, module_build_dir):
    options = []
    if self.module_dir_switch is not None:
        if self.module_dir_switch[-1] == ' ':
            options.extend([self.module_dir_switch.strip(), module_build_dir])
        else:
            options.append(self.module_dir_switch.strip() + module_build_dir)
    else:
        print('XXX: module_build_dir=%r option ignored' % module_build_dir)
        print('XXX: Fix module_dir_switch for ', self.__class__.__name__)
    if self.module_include_switch is not None:
        for d in [module_build_dir] + module_dirs:
            options.append('%s%s' % (self.module_include_switch, d))
    else:
        print('XXX: module_dirs=%r option ignored' % module_dirs)
        print('XXX: Fix module_include_switch for ', self.__class__.__name__)
    return options