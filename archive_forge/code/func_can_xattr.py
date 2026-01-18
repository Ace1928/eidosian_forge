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
def can_xattr():
    global _can_xattr
    if _can_xattr is not None:
        return _can_xattr
    if not hasattr(os, 'setxattr'):
        can = False
    else:
        tmp_fp, tmp_name = tempfile.mkstemp()
        try:
            with open(TESTFN, 'wb') as fp:
                try:
                    os.setxattr(tmp_fp, b'user.test', b'')
                    os.setxattr(fp.fileno(), b'user.test', b'')
                    kernel_version = platform.release()
                    m = re.match('2.6.(\\d{1,2})', kernel_version)
                    can = m is None or int(m.group(1)) >= 39
                except OSError:
                    can = False
        finally:
            unlink(TESTFN)
            unlink(tmp_name)
    _can_xattr = can
    return can