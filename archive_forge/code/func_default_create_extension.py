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
def default_create_extension(template, kwds):
    if 'depends' in kwds:
        include_dirs = kwds.get('include_dirs', []) + ['.']
        depends = resolve_depends(kwds['depends'], include_dirs)
        kwds['depends'] = sorted(set(depends + template.depends))
    t = template.__class__
    ext = t(**kwds)
    metadata = dict(distutils=kwds, module_name=kwds['name'])
    return (ext, metadata)