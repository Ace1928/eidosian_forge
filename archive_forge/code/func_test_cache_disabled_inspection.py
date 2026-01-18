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
def test_cache_disabled_inspection(self):
    """
        """
    library = self.compile_module(asm_sum_outer, asm_sum_inner)
    library.enable_object_caching()
    state = library.serialize_using_object_code()
    with warnings.catch_warnings(record=True) as w:
        old_llvm = library.get_llvm_str()
        old_asm = library.get_asm_str()
        library.get_function_cfg('sum')
    self.assertEqual(len(w), 0)
    codegen = JITCPUCodegen('other_codegen')
    library = codegen.unserialize_library(state)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        self.assertNotEqual(old_llvm, library.get_llvm_str())
    self.assertEqual(len(w), 1)
    self.assertIn('Inspection disabled', str(w[0].message))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        self.assertNotEqual(library.get_asm_str(), old_asm)
    self.assertEqual(len(w), 1)
    self.assertIn('Inspection disabled', str(w[0].message))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with self.assertRaises(NameError) as raises:
            library.get_function_cfg('sum')
    self.assertEqual(len(w), 1)
    self.assertIn('Inspection disabled', str(w[0].message))
    self.assertIn('sum', str(raises.exception))