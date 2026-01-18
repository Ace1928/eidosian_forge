from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
def distutils_info0(self, filename):
    info = self.parse_dependencies(filename)[3]
    kwds = info.values
    cimports, externs, incdirs = self.cimports_externs_incdirs(filename)
    basedir = os.getcwd()
    if externs:
        externs = _make_relative(externs, basedir)
        if 'depends' in kwds:
            kwds['depends'] = list(set(kwds['depends']).union(externs))
        else:
            kwds['depends'] = list(externs)
    if incdirs:
        include_dirs = list(kwds.get('include_dirs', []))
        for inc in _make_relative(incdirs, basedir):
            if inc not in include_dirs:
                include_dirs.append(inc)
        kwds['include_dirs'] = include_dirs
    return info