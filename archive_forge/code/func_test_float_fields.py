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
def test_float_fields(self):

    class S(cgutils.Structure):
        _fields = [('a', types.float64), ('b', types.float32)]
    fmt = '=df'
    with self.run_simple_struct_test(S, fmt, (1.23, 4.56)) as (context, builder, inst):
        inst.a = ir.Constant(ir.DoubleType(), 1.23)
        inst.b = ir.Constant(ir.FloatType(), 4.56)