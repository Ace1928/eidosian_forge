import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
def no_de_locale():
    cur = locale.setlocale(locale.LC_ALL)
    try:
        locale.setlocale(locale.LC_ALL, 'de_DE')
    except locale.Error:
        return True
    else:
        return False
    finally:
        locale.setlocale(locale.LC_ALL, cur)