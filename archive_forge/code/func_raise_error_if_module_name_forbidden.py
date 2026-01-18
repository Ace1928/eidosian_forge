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
def raise_error_if_module_name_forbidden(full_module_name):
    if full_module_name == 'cython' or full_module_name.startswith('cython.'):
        raise ValueError('cython is a special module, cannot be used as a module name')