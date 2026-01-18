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
def dump_properties(self):
    """Print out the attributes of a compiler instance."""
    props = []
    for key in list(self.executables.keys()) + ['version', 'libraries', 'library_dirs', 'object_switch', 'compile_switch']:
        if hasattr(self, key):
            v = getattr(self, key)
            props.append((key, None, '= ' + repr(v)))
    props.sort()
    pretty_printer = FancyGetopt(props)
    for l in pretty_printer.generate_help('%s instance properties:' % self.__class__.__name__):
        if l[:4] == '  --':
            l = '  ' + l[4:]
        print(l)