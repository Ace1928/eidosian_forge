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
def test_jumps(self):
    code = Bytecode()
    label_else = Label()
    label_return = Label()
    code.extend([Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label_else), Instr('LOAD_CONST', 12, lineno=2), Instr('STORE_NAME', 'x'), Instr('JUMP_FORWARD', label_return), label_else, Instr('LOAD_CONST', 37, lineno=4), Instr('STORE_NAME', 'x'), label_return, Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE')])
    code = code.to_concrete_bytecode()
    expected = [ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('POP_JUMP_IF_FALSE', 5 if OFFSET_AS_INSTRUCTION else 10, lineno=1), ConcreteInstr('LOAD_CONST', 0, lineno=2), ConcreteInstr('STORE_NAME', 1, lineno=2), ConcreteInstr('JUMP_FORWARD', 2 if OFFSET_AS_INSTRUCTION else 4, lineno=2), ConcreteInstr('LOAD_CONST', 1, lineno=4), ConcreteInstr('STORE_NAME', 1, lineno=4), ConcreteInstr('LOAD_CONST', 2, lineno=4), ConcreteInstr('RETURN_VALUE', lineno=4)]
    self.assertListEqual(list(code), expected)
    self.assertListEqual(code.consts, [12, 37, None])
    self.assertListEqual(code.names, ['test', 'x'])
    self.assertListEqual(code.varnames, [])