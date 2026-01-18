from __future__ import absolute_import
import hashlib
import inspect
import os
import re
import sys
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
import Cython
from ..Compiler.Main import Context
from ..Compiler.Options import (default_options, CompilationOptions,
from ..Compiler.Visitor import CythonTransform, EnvTransform
from ..Compiler.ParseTreeTransforms import SkipDeclarations
from ..Compiler.TreeFragment import parse_from_strings
from ..Compiler.StringEncoding import _unicode
from .Dependencies import strip_string_literals, cythonize, cached_function
from ..Compiler import Pipeline
from ..Utils import get_cython_cache_dir
import cython as cython_module
def safe_type(arg, context=None):
    py_type = type(arg)
    if py_type in (list, tuple, dict, str):
        return py_type.__name__
    elif py_type is complex:
        return 'double complex'
    elif py_type is float:
        return 'double'
    elif py_type is bool:
        return 'bint'
    elif 'numpy' in sys.modules and isinstance(arg, sys.modules['numpy'].ndarray):
        return 'numpy.ndarray[numpy.%s_t, ndim=%s]' % (arg.dtype.name, arg.ndim)
    else:
        for base_type in py_type.__mro__:
            if base_type.__module__ in ('__builtin__', 'builtins'):
                return 'object'
            module = context.find_module(base_type.__module__, need_pxd=False)
            if module:
                entry = module.lookup(base_type.__name__)
                if entry.is_type:
                    return '%s.%s' % (base_type.__module__, base_type.__name__)
        return 'object'