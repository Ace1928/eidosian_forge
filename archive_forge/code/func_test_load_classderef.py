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
def test_load_classderef(self):
    concrete = ConcreteBytecode()
    concrete.cellvars = ['__class__']
    concrete.freevars = ['__class__']
    concrete.extend([ConcreteInstr('LOAD_CLASSDEREF', 1), ConcreteInstr('STORE_DEREF', 1)])
    bytecode = concrete.to_bytecode()
    self.assertEqual(bytecode.freevars, ['__class__'])
    self.assertEqual(bytecode.cellvars, ['__class__'])
    self.assertEqual(list(bytecode), [Instr('LOAD_CLASSDEREF', FreeVar('__class__'), lineno=1), Instr('STORE_DEREF', FreeVar('__class__'), lineno=1)])
    concrete = bytecode.to_concrete_bytecode()
    self.assertEqual(concrete.freevars, ['__class__'])
    self.assertEqual(concrete.cellvars, ['__class__'])
    self.assertEqual(list(concrete), [ConcreteInstr('LOAD_CLASSDEREF', 1, lineno=1), ConcreteInstr('STORE_DEREF', 1, lineno=1)])
    code = concrete.to_code()
    self.assertEqual(code.co_freevars, ('__class__',))
    self.assertEqual(code.co_cellvars, ('__class__',))
    self.assertEqual(code.co_code, b'\x94\x01\x89\x01')