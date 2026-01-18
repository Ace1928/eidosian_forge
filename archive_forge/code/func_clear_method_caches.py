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
def clear_method_caches(obj):
    """Removes every cache found in the object,
    if a corresponding method exists for that cache.
    """
    for cache_name, method_name in _find_cache_attributes(obj):
        if hasattr(obj, method_name):
            delattr(obj, cache_name)