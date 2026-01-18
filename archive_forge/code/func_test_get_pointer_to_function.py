import warnings
import base64
import ctypes
import pickle
import re
import subprocess
import sys
import weakref
import llvmlite.binding as ll
import unittest
from numba import njit
from numba.core.codegen import JITCPUCodegen
from numba.core.compiler_lock import global_compiler_lock
from numba.tests.support import TestCase
def test_get_pointer_to_function(self):
    library = self.compile_module(asm_sum)
    ptr = library.get_pointer_to_function('sum')
    self.assertIsInstance(ptr, int)
    cfunc = ctypes_sum_ty(ptr)
    self.assertEqual(cfunc(2, 3), 5)
    library2 = self.compile_module(asm_sum_outer, asm_sum_inner)
    ptr = library2.get_pointer_to_function('sum')
    self.assertIsInstance(ptr, int)
    cfunc = ctypes_sum_ty(ptr)
    self.assertEqual(cfunc(2, 3), 5)