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
def check_pid(pid):
    return bool(ctypes.windll.kernel32.OpenProcess(1, 0, pid))