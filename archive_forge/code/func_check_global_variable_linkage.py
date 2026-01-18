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
def check_global_variable_linkage(self, linkage, has_undef=True):
    mod = ir.Module()
    typ = ir.IntType(32)
    gv = ir.GlobalVariable(mod, typ, 'foo')
    gv.linkage = linkage
    asm = str(mod)
    if has_undef:
        self.assertIn('undef', asm)
    else:
        self.assertNotIn('undef', asm)
    self.module(asm)