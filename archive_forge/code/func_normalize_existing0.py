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
def normalize_existing0(base_dir, rel_paths):
    """
    Given some base directory ``base_dir`` and a list of path names
    ``rel_paths``, normalize each relative path name ``rel`` by
    replacing it by ``os.path.join(base, rel)`` if that file exists.

    Return a couple ``(normalized, needed_base)`` where ``normalized``
    if the list of normalized file names and ``needed_base`` is
    ``base_dir`` if we actually needed ``base_dir``. If no paths were
    changed (for example, if all paths were already absolute), then
    ``needed_base`` is ``None``.
    """
    normalized = []
    needed_base = None
    for rel in rel_paths:
        if os.path.isabs(rel):
            normalized.append(rel)
            continue
        path = join_path(base_dir, rel)
        if path_exists(path):
            normalized.append(os.path.normpath(path))
            needed_base = base_dir
        else:
            normalized.append(rel)
    return (normalized, needed_base)