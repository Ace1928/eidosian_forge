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
def test_nonalphanum_block_name(self):
    mod = ir.Module()
    ft = ir.FunctionType(ir.IntType(32), [])
    fn = ir.Function(mod, ft, 'foo')
    bd = ir.IRBuilder(fn.append_basic_block(name="<>!*''#"))
    bd.ret(ir.Constant(ir.IntType(32), 12345))
    asm = str(mod)
    self.assertEqual(asm, asm_nonalphanum_blocklabel)