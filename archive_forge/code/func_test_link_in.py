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
def test_link_in(self):
    dest = self.module()
    src = self.module(asm_mul)
    dest.link_in(src)
    self.assertEqual(sorted((f.name for f in dest.functions)), ['mul', 'sum'])
    dest.get_function('mul')
    dest.close()
    with self.assertRaises(ctypes.ArgumentError):
        src.get_function('mul')