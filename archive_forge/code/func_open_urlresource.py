from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
def open_urlresource(url, *args, **kw):
    from future.backports.urllib import request as urllib_request, parse as urllib_parse
    check = kw.pop('check', None)
    filename = urllib_parse.urlparse(url)[2].split('/')[-1]
    fn = os.path.join(os.path.dirname(__file__), 'data', filename)

    def check_valid_file(fn):
        f = open(fn, *args, **kw)
        if check is None:
            return f
        elif check(f):
            f.seek(0)
            return f
        f.close()
    if os.path.exists(fn):
        f = check_valid_file(fn)
        if f is not None:
            return f
        unlink(fn)
    requires('urlfetch')
    print('\tfetching %s ...' % url, file=get_original_stdout())
    f = urllib_request.urlopen(url, timeout=15)
    try:
        with open(fn, 'wb') as out:
            s = f.read()
            while s:
                out.write(s)
                s = f.read()
    finally:
        f.close()
    f = check_valid_file(fn)
    if f is not None:
        return f
    raise TestFailed('invalid resource %r' % fn)