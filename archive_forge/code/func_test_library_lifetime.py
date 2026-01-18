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
@unittest.expectedFailure
def test_library_lifetime(self):
    library = self.compile_module(asm_sum_outer, asm_sum_inner)
    library.enable_object_caching()
    library.serialize_using_bitcode()
    library.serialize_using_object_code()
    u = weakref.ref(library)
    v = weakref.ref(library._final_module)
    del library
    self.assertIs(u(), None)
    self.assertIs(v(), None)