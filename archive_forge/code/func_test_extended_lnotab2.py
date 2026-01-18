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
def test_extended_lnotab2(self):
    base_code = compile('x = 7' + '\n' * 200 + 'y = 8', '', 'exec')
    concrete = ConcreteBytecode([ConcreteInstr('LOAD_CONST', 0), ConcreteInstr('STORE_NAME', 0), SetLineno(201), ConcreteInstr('LOAD_CONST', 1), ConcreteInstr('STORE_NAME', 1), ConcreteInstr('LOAD_CONST', 2), ConcreteInstr('RETURN_VALUE')])
    concrete.consts = [None, 7, 8]
    concrete.names = ['x', 'y']
    concrete.first_lineno = 1
    code = concrete.to_code()
    self.assertEqual(code.co_code, base_code.co_code)
    self.assertEqual(code.co_firstlineno, base_code.co_firstlineno)
    self.assertEqual(code.co_lnotab, base_code.co_lnotab)
    if sys.version_info >= (3, 10):
        self.assertEqual(code.co_linetable, base_code.co_linetable)