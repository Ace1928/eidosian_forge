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
def cached_function(f):
    cache = {}
    _function_caches.append(cache)
    uncomputed = object()

    @wraps(f)
    def wrapper(*args):
        res = cache.get(args, uncomputed)
        if res is uncomputed:
            res = cache[args] = f(*args)
        return res
    wrapper.uncached = f
    return wrapper