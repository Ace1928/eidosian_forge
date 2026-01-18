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
def cleanup_cache(cache, target_size, ratio=0.85):
    try:
        p = subprocess.Popen(['du', '-s', '-k', os.path.abspath(cache)], stdout=subprocess.PIPE)
        stdout, _ = p.communicate()
        res = p.wait()
        if res == 0:
            total_size = 1024 * int(stdout.strip().split()[0])
            if total_size < target_size:
                return
    except (OSError, ValueError):
        pass
    total_size = 0
    all = []
    for file in os.listdir(cache):
        path = join_path(cache, file)
        s = os.stat(path)
        total_size += s.st_size
        all.append((s.st_atime, s.st_size, path))
    if total_size > target_size:
        for time, size, file in reversed(sorted(all)):
            os.unlink(file)
            total_size -= size
            if total_size < target_size * ratio:
                break