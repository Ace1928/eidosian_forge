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
@contextlib.contextmanager
def run_simple_struct_test(self, struct_class, struct_fmt, struct_args):
    buf = bytearray(b'!') * 40
    expected = buf[:]
    offset = 8
    with self.run_struct_access(struct_class, buf, offset) as (context, builder, args, inst):
        yield (context, builder, inst)
    self.assertNotEqual(buf, expected)
    struct.pack_into(struct_fmt, expected, offset, *struct_args)
    self.assertEqual(buf, expected)