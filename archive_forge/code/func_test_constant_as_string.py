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
def test_constant_as_string(self):
    mod = self.module(asm_null_constant)
    func = mod.get_function('bar')
    inst = list(list(func.blocks)[0].instructions)[0]
    arg = list(inst.operands)[0]
    self.assertTrue(arg.is_constant)
    self.assertEqual(arg.get_constant_value(), 'i64* null')