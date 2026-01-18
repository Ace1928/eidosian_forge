import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
def test_from_code(self):
    code = get_code('\n            if test:\n                x = 1\n            else:\n                x = 2\n        ')
    bytecode = Bytecode.from_code(code)
    label_else = Label()
    label_exit = Label()
    if sys.version_info < (3, 10):
        self.assertEqual(bytecode, [Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label_else, lineno=1), Instr('LOAD_CONST', 1, lineno=2), Instr('STORE_NAME', 'x', lineno=2), Instr('JUMP_FORWARD', label_exit, lineno=2), label_else, Instr('LOAD_CONST', 2, lineno=4), Instr('STORE_NAME', 'x', lineno=4), label_exit, Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)])
    else:
        self.assertEqual(bytecode, [Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label_else, lineno=1), Instr('LOAD_CONST', 1, lineno=2), Instr('STORE_NAME', 'x', lineno=2), Instr('LOAD_CONST', None, lineno=2), Instr('RETURN_VALUE', lineno=2), label_else, Instr('LOAD_CONST', 2, lineno=4), Instr('STORE_NAME', 'x', lineno=4), Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)])