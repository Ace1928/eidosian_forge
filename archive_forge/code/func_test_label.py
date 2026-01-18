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
def test_label(self):
    code = Bytecode()
    label = Label()
    code.extend([Instr('LOAD_CONST', 'hello', lineno=1), Instr('JUMP_FORWARD', label, lineno=1), label, Instr('POP_TOP', lineno=1)])
    code = code.to_concrete_bytecode()
    expected = [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('JUMP_FORWARD', 0, lineno=1), ConcreteInstr('POP_TOP', lineno=1)]
    self.assertListEqual(list(code), expected)
    self.assertListEqual(code.consts, ['hello'])