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
@cached_function
def resolve_depend(depend, include_dirs):
    if depend[0] == '<' and depend[-1] == '>':
        return None
    for dir in include_dirs:
        path = join_path(dir, depend)
        if path_exists(path):
            return os.path.normpath(path)
    return None