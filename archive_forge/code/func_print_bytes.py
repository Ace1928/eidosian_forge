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
def print_bytes(s, header_text=None, end=b'\n', file=sys.stdout, flush=True):
    if header_text:
        file.write(header_text)
    file.flush()
    try:
        out = file.buffer
    except AttributeError:
        out = file
    out.write(s)
    if end:
        out.write(end)
    if flush:
        out.flush()