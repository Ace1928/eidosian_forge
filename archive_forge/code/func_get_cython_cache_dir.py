from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
@cached_function
def get_cython_cache_dir():
    """
    Return the base directory containing Cython's caches.

    Priority:

    1. CYTHON_CACHE_DIR
    2. (OS X): ~/Library/Caches/Cython
       (posix not OS X): XDG_CACHE_HOME/cython if XDG_CACHE_HOME defined
    3. ~/.cython

    """
    if 'CYTHON_CACHE_DIR' in os.environ:
        return os.environ['CYTHON_CACHE_DIR']
    parent = None
    if os.name == 'posix':
        if sys.platform == 'darwin':
            parent = os.path.expanduser('~/Library/Caches')
        else:
            parent = os.environ.get('XDG_CACHE_HOME')
    if parent and os.path.isdir(parent):
        return os.path.join(parent, 'cython')
    return os.path.expanduser(os.path.join('~', '.cython'))