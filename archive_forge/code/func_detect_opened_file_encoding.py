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
def detect_opened_file_encoding(f, default='UTF-8'):
    lines = ()
    start = b''
    while len(lines) < 3:
        data = f.read(500)
        start += data
        lines = start.split(b'\n')
        if not data:
            break
    m = _match_file_encoding(lines[0])
    if m and m.group(1) != b'c_string_encoding':
        return m.group(2).decode('iso8859-1')
    elif len(lines) > 1:
        m = _match_file_encoding(lines[1])
        if m:
            return m.group(2).decode('iso8859-1')
    return default