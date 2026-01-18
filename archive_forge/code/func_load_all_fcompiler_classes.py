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
def load_all_fcompiler_classes():
    """Cache all the FCompiler classes found in modules in the
    numpy.distutils.fcompiler package.
    """
    from glob import glob
    global fcompiler_class, fcompiler_aliases
    if fcompiler_class is not None:
        return
    pys = os.path.join(os.path.dirname(__file__), '*.py')
    fcompiler_class = {}
    fcompiler_aliases = {}
    for fname in glob(pys):
        module_name, ext = os.path.splitext(os.path.basename(fname))
        module_name = 'numpy.distutils.fcompiler.' + module_name
        __import__(module_name)
        module = sys.modules[module_name]
        if hasattr(module, 'compilers'):
            for cname in module.compilers:
                klass = getattr(module, cname)
                desc = (klass.compiler_type, klass, klass.description)
                fcompiler_class[klass.compiler_type] = desc
                for alias in klass.compiler_aliases:
                    if alias in fcompiler_aliases:
                        raise ValueError('alias %r defined for both %s and %s' % (alias, klass.__name__, fcompiler_aliases[alias][1].__name__))
                    fcompiler_aliases[alias] = desc