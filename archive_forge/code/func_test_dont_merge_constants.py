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
def test_dont_merge_constants(self):
    code = Bytecode()
    code.extend([Instr('LOAD_CONST', 5, lineno=1), Instr('LOAD_CONST', 5.0, lineno=1), Instr('LOAD_CONST', -0.0, lineno=1), Instr('LOAD_CONST', +0.0, lineno=1)])
    code = code.to_concrete_bytecode()
    expected = [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('LOAD_CONST', 1, lineno=1), ConcreteInstr('LOAD_CONST', 2, lineno=1), ConcreteInstr('LOAD_CONST', 3, lineno=1)]
    self.assertListEqual(list(code), expected)
    self.assertListEqual(code.consts, [5, 5.0, -0.0, +0.0])