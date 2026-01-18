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
def build_ir_module(self):
    m = ir.Module()
    ft = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(32)])
    fn = ir.Function(m, ft, 'foo')
    bd = ir.IRBuilder(fn.append_basic_block())
    x, y = fn.args
    z = bd.add(x, y)
    bd.ret(z)
    return m