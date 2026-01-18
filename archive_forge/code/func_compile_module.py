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
def compile_module(self, asm, linking_asm=None):
    library = self.codegen.create_library('compiled_module')
    ll_module = ll.parse_assembly(asm)
    ll_module.verify()
    library.add_llvm_module(ll_module)
    if linking_asm:
        linking_library = self.codegen.create_library('linking_module')
        ll_module = ll.parse_assembly(linking_asm)
        ll_module.verify()
        linking_library.add_llvm_module(ll_module)
        library.add_linking_library(linking_library)
    return library