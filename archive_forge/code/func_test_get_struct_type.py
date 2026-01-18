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
def test_get_struct_type(self):
    mod = self.module()
    st_ty = mod.get_struct_type('struct.glob_type')
    self.assertEqual(st_ty.name, 'struct.glob_type')
    self.assertIsNotNone(re.match('%struct\\.glob_type(\\.[\\d]+)? = type { i64, \\[2 x i64\\] }', str(st_ty)))
    with self.assertRaises(NameError):
        mod.get_struct_type('struct.doesnt_exist')