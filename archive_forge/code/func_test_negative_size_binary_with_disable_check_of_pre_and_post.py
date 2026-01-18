import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
def test_negative_size_binary_with_disable_check_of_pre_and_post(self):
    opnames = ('BINARY_POWER', 'BINARY_MULTIPLY', 'BINARY_MATRIX_MULTIPLY', 'BINARY_FLOOR_DIVIDE', 'BINARY_TRUE_DIVIDE', 'BINARY_MODULO', 'BINARY_ADD', 'BINARY_SUBTRACT', 'BINARY_SUBSCR', 'BINARY_LSHIFT', 'BINARY_RSHIFT', 'BINARY_AND', 'BINARY_XOR', 'BINARY_OR')
    for opname in opnames:
        with self.subTest():
            code = Bytecode()
            code.first_lineno = 1
            code.extend([Instr('LOAD_CONST', 1), Instr(opname)])
            co = code.to_code(check_pre_and_post=False)
            self.assertEqual(co.co_stacksize, 1)