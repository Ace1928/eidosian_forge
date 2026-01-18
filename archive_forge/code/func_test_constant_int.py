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
def test_constant_int(self):
    mod = self.module()
    func = mod.get_function('sum')
    insts = list(list(func.blocks)[0].instructions)
    self.assertEqual(insts[1].opcode, 'add')
    operands = list(insts[1].operands)
    self.assertTrue(operands[0].is_constant)
    self.assertFalse(operands[1].is_constant)
    self.assertEqual(operands[0].get_constant_value(), 0)
    with self.assertRaises(ValueError):
        operands[1].get_constant_value()
    mod = self.module(asm_sum3)
    func = mod.get_function('sum')
    insts = list(list(func.blocks)[0].instructions)
    posint64 = list(insts[1].operands)[0]
    negint64 = list(insts[2].operands)[0]
    self.assertEqual(posint64.get_constant_value(), 5)
    self.assertEqual(negint64.get_constant_value(signed_int=True), -5)
    as_u64 = negint64.get_constant_value(signed_int=False)
    as_i64 = int.from_bytes(as_u64.to_bytes(8, 'little'), 'little', signed=True)
    self.assertEqual(as_i64, -5)