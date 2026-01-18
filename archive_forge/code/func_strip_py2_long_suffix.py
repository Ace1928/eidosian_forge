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
def strip_py2_long_suffix(value_str):
    """
    Python 2 likes to append 'L' to stringified numbers
    which in then can't process when converting them to numbers.
    """
    if value_str[-1] in 'lL':
        return value_str[:-1]
    return value_str