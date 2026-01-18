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
def extract_func_code(code):
    module = []
    function = []
    current = function
    code = code.replace('\t', ' ')
    lines = code.split('\n')
    for line in lines:
        if not line.startswith(' '):
            if module_statement.match(line):
                current = module
            else:
                current = function
        current.append(line)
    return ('\n'.join(module), '    ' + '\n    '.join(function))