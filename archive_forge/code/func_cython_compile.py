from __future__ import absolute_import, print_function
import os
import shutil
import tempfile
from .Dependencies import cythonize, extended_iglob
from ..Utils import is_package_dir
from ..Compiler import Options
def cython_compile(path_pattern, options):
    all_paths = map(os.path.abspath, extended_iglob(path_pattern))
    _cython_compile_files(all_paths, options)