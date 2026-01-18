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
def test_freevar(self):
    concrete = ConcreteBytecode()
    concrete.freevars = ['x']
    concrete.append(ConcreteInstr('LOAD_DEREF', 0))
    code = concrete.to_code()
    concrete = ConcreteBytecode.from_code(code)
    self.assertEqual(concrete.cellvars, [])
    self.assertEqual(concrete.freevars, ['x'])
    self.assertEqual(list(concrete), [ConcreteInstr('LOAD_DEREF', 0, lineno=1)])
    bytecode = concrete.to_bytecode()
    self.assertEqual(bytecode.cellvars, [])
    self.assertEqual(list(bytecode), [Instr('LOAD_DEREF', FreeVar('x'), lineno=1)])