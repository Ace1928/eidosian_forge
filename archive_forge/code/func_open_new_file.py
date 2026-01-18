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
def open_new_file(path):
    if os.path.exists(path):
        os.unlink(path)
    return codecs.open(path, 'w', encoding='ISO-8859-1')