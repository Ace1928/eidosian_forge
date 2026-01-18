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
def contains_init(dir_path):
    for filename in PACKAGE_FILES:
        path = os.path.join(dir_path, filename)
        if path_exists(path):
            return 1