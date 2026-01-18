import os
import sys
import ctypes
import time
from ctypes import c_int, POINTER
from ctypes.wintypes import LPCWSTR, HLOCAL
from subprocess import STDOUT, TimeoutExpired
from threading import Thread
from ._process_common import read_no_interrupt, process_handler, arg_split as py_arg_split
from . import py3compat
from .encoding import DEFAULT_ENCODING
def stderr_read():
    for line in read_no_interrupt(p.stderr).splitlines():
        line = line.decode(enc, 'replace')
        print(line, file=sys.stderr)