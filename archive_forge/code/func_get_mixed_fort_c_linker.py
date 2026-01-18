import glob
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from sysconfig import get_config_var, get_config_vars, get_path
from .runners import (
from .util import (
def get_mixed_fort_c_linker(vendor=None, cplus=False, cwd=None):
    vendor = vendor or os.environ.get('SYMPY_COMPILER_VENDOR', 'gnu')
    if vendor.lower() == 'intel':
        if cplus:
            return (FortranCompilerRunner, {'flags': ['-nofor_main', '-cxxlib']}, vendor)
        else:
            return (FortranCompilerRunner, {'flags': ['-nofor_main']}, vendor)
    elif vendor.lower() == 'gnu' or 'llvm':
        if cplus:
            return (CppCompilerRunner, {'lib_options': ['fortran']}, vendor)
        else:
            return (FortranCompilerRunner, {}, vendor)
    else:
        raise ValueError('No vendor found.')