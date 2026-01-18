import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def get_module_file(module_name):
    """Return the generated module file based on module name."""
    path = os.path.dirname(__file__)
    module_path = module_name.split('.')
    module_path[-1] = 'gen_' + module_path[-1]
    file_name = os.path.join(path, '..', *module_path) + '.py'
    module_file = open(file_name, 'w', encoding='utf-8')
    dependencies = {'symbol': ['from ._internal import SymbolBase', 'from ..base import _Null'], 'ndarray': ['from ._internal import NDArrayBase', 'from ..base import _Null']}
    module_file.write('# coding: utf-8')
    module_file.write(license_str)
    module_file.write('# File content is auto-generated. Do not modify.' + os.linesep)
    module_file.write('# pylint: skip-file' + os.linesep)
    module_file.write(os.linesep.join(dependencies[module_name.split('.')[1]]))
    return module_file