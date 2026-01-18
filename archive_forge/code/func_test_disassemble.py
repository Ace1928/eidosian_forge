import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import sys
import textwrap
import types
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import get_code, TestCase
def test_disassemble(self):
    code = b'\t\x00d\x03'
    instr = ConcreteInstr.disassemble(1, code, 0)
    self.assertEqual(instr, ConcreteInstr('NOP', lineno=1))
    instr = ConcreteInstr.disassemble(2, code, 1 if OFFSET_AS_INSTRUCTION else 2)
    self.assertEqual(instr, ConcreteInstr('LOAD_CONST', 3, lineno=2))
    code = b'\x90\x12\x904\x90\xabd\xcd'
    instr = ConcreteInstr.disassemble(3, code, 0)
    self.assertEqual(instr, ConcreteInstr('EXTENDED_ARG', 18, lineno=3))