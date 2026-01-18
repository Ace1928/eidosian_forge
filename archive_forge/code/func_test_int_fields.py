import contextlib
import ctypes
import struct
import sys
import llvmlite.ir as ir
import numpy as np
import unittest
from numba.core import types, typing, cgutils, cpu
from numba.core.compiler_lock import global_compiler_lock
from numba.tests.support import TestCase, run_in_subprocess
def test_int_fields(self):

    class S(cgutils.Structure):
        _fields = [('a', types.int32), ('b', types.uint16)]
    fmt = '=iH'
    with self.run_simple_struct_test(S, fmt, (305419896, 43981)) as (context, builder, inst):
        inst.a = ir.Constant(ir.IntType(32), 305419896)
        inst.b = ir.Constant(ir.IntType(16), 43981)