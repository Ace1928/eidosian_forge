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
def test_get_function_cfg_on_ir(self):
    mod = self.build_ir_module()
    foo = mod.get_global('foo')
    dot_showing_inst = llvm.get_function_cfg(foo)
    dot_without_inst = llvm.get_function_cfg(foo, show_inst=False)
    inst = '%.5 = add i32 %.1, %.2'
    self.assertIn(inst, dot_showing_inst)
    self.assertNotIn(inst, dot_without_inst)