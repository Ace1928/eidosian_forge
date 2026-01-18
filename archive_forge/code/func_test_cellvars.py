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
def test_cellvars(self):
    code = Bytecode()
    code.cellvars = ['x']
    code.freevars = ['y']
    code.extend([Instr('LOAD_DEREF', CellVar('x'), lineno=1), Instr('LOAD_DEREF', FreeVar('y'), lineno=1)])
    concrete = code.to_concrete_bytecode()
    self.assertEqual(concrete.cellvars, ['x'])
    self.assertEqual(concrete.freevars, ['y'])
    code.extend([ConcreteInstr('LOAD_DEREF', 0, lineno=1), ConcreteInstr('LOAD_DEREF', 1, lineno=1)])