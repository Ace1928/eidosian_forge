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
def test_instruction_operands(self):
    func = self.module().get_function('sum')
    add = list(list(func.blocks)[0].instructions)[0]
    self.assertEqual(add.opcode, 'add')
    operands = list(add.operands)
    self.assertEqual(len(operands), 2)
    self.assertTrue(operands[0].is_operand)
    self.assertTrue(operands[1].is_operand)
    self.assertEqual(operands[0].name, '.1')
    self.assertEqual(str(operands[0].type), 'i32')
    self.assertEqual(operands[1].name, '.2')
    self.assertEqual(str(operands[1].type), 'i32')