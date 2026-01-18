import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
def test_async_gen_no_flag_is_async_None(self):
    code = ConcreteBytecode()
    code.append(ConcreteInstr('YIELD_VALUE'))
    code.update_flags()
    self.assertTrue(bool(code.flags & CompilerFlags.GENERATOR))
    code = ConcreteBytecode()
    code.append(ConcreteInstr('GET_AWAITABLE'))
    code.update_flags()
    self.assertTrue(bool(code.flags & CompilerFlags.COROUTINE))
    for i, expected in (('YIELD_VALUE', CompilerFlags.ASYNC_GENERATOR), ('YIELD_FROM', CompilerFlags.COROUTINE)):
        code = ConcreteBytecode()
        code.append(ConcreteInstr('GET_AWAITABLE'))
        code.append(ConcreteInstr(i))
        code.update_flags()
        self.assertTrue(bool(code.flags & expected))