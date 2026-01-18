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
def test_module_functions(self):
    mod = self.module()
    funcs = list(mod.functions)
    self.assertEqual(len(funcs), 1)
    func = funcs[0]
    self.assertTrue(func.is_function)
    self.assertEqual(func.name, 'sum')
    with self.assertRaises(ValueError):
        func.instructions
    with self.assertRaises(ValueError):
        func.operands
    with self.assertRaises(ValueError):
        func.opcode