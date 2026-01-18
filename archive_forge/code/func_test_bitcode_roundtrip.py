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
def test_bitcode_roundtrip(self):
    context1 = llvm.create_context()
    bc = self.module(context=context1).as_bitcode()
    context2 = llvm.create_context()
    mod = llvm.parse_bitcode(bc, context2)
    self.assertEqual(mod.as_bitcode(), bc)
    mod.get_function('sum')
    mod.get_global_variable('glob')