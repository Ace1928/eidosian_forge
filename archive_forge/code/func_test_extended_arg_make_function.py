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
def test_extended_arg_make_function(self):
    if (3, 9) <= sys.version_info < (3, 10):
        from _pydevd_frame_eval.vendored.bytecode.tests.util_annotation import get_code as get_code_future
        code_obj = get_code_future('\n                def foo(x: int, y: int):\n                    pass\n                ')
    else:
        code_obj = get_code('\n                def foo(x: int, y: int):\n                    pass\n                ')
    concrete = ConcreteBytecode.from_code(code_obj)
    if sys.version_info >= (3, 10):
        func_code = concrete.consts[2]
        names = ['int', 'foo']
        consts = ['x', 'y', func_code, 'foo', None]
        const_offset = 1
        name_offset = 1
        first_instrs = [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('LOAD_CONST', 1, lineno=1), ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('BUILD_TUPLE', 4, lineno=1)]
    elif sys.version_info >= (3, 7) and concrete.flags & CompilerFlags.FUTURE_ANNOTATIONS:
        func_code = concrete.consts[2]
        names = ['foo']
        consts = ['int', ('x', 'y'), func_code, 'foo', None]
        const_offset = 1
        name_offset = 0
        first_instrs = [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('LOAD_CONST', 0 + const_offset, lineno=1), ConcreteInstr('BUILD_CONST_KEY_MAP', 2, lineno=1)]
    else:
        func_code = concrete.consts[1]
        names = ['int', 'foo']
        consts = [('x', 'y'), func_code, 'foo', None]
        const_offset = 0
        name_offset = 1
        first_instrs = [ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('LOAD_CONST', 0 + const_offset, lineno=1), ConcreteInstr('BUILD_CONST_KEY_MAP', 2, lineno=1)]
    self.assertEqual(concrete.names, names)
    self.assertEqual(concrete.consts, consts)
    expected = first_instrs + [ConcreteInstr('LOAD_CONST', 1 + const_offset, lineno=1), ConcreteInstr('LOAD_CONST', 2 + const_offset, lineno=1), ConcreteInstr('MAKE_FUNCTION', 4, lineno=1), ConcreteInstr('STORE_NAME', name_offset, lineno=1), ConcreteInstr('LOAD_CONST', 3 + const_offset, lineno=1), ConcreteInstr('RETURN_VALUE', lineno=1)]
    self.assertListEqual(list(concrete), expected)
    concrete = ConcreteBytecode.from_code(code_obj, extended_arg=True)
    if sys.version_info >= (3, 10):
        func_code = concrete.consts[2]
        names = ['int', 'foo']
        consts = ['x', 'y', func_code, 'foo', None]
    elif concrete.flags & CompilerFlags.FUTURE_ANNOTATIONS:
        func_code = concrete.consts[2]
        names = ['foo']
        consts = ['int', ('x', 'y'), func_code, 'foo', None]
    else:
        func_code = concrete.consts[1]
        names = ['int', 'foo']
        consts = [('x', 'y'), func_code, 'foo', None]
    self.assertEqual(concrete.names, names)
    self.assertEqual(concrete.consts, consts)
    self.assertListEqual(list(concrete), expected)