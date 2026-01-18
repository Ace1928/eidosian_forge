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
def get_encoding_candidates():
    candidates = [sys.getdefaultencoding()]
    for stream in (sys.stdout, sys.stdin, sys.__stdout__, sys.__stdin__):
        encoding = getattr(stream, 'encoding', None)
        if encoding is not None and encoding not in candidates:
            candidates.append(encoding)
    return candidates