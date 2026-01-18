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
def test_constant_fp(self):
    mod = self.module(asm_double_locale)
    func = mod.get_function('foo')
    insts = list(list(func.blocks)[0].instructions)
    self.assertEqual(len(insts), 2)
    self.assertEqual(insts[0].opcode, 'fadd')
    operands = list(insts[0].operands)
    self.assertTrue(operands[0].is_constant)
    self.assertAlmostEqual(operands[0].get_constant_value(), 0.0)
    self.assertTrue(operands[1].is_constant)
    self.assertAlmostEqual(operands[1].get_constant_value(), 3.14)
    mod = self.module(asm_double_inaccurate)
    func = mod.get_function('foo')
    inst = list(list(func.blocks)[0].instructions)[0]
    operands = list(inst.operands)
    with self.assertRaises(ValueError):
        operands[0].get_constant_value()
    self.assertAlmostEqual(operands[1].get_constant_value(round_fp=True), 0)