import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
def test_flag_inference(self):
    code = ControlFlowGraph()
    code.flags |= CompilerFlags.NEWLOCALS | CompilerFlags.VARARGS | CompilerFlags.VARKEYWORDS | CompilerFlags.NESTED | CompilerFlags.FUTURE_GENERATOR_STOP
    code.update_flags()
    for f in (CompilerFlags.NEWLOCALS, CompilerFlags.VARARGS, CompilerFlags.VARKEYWORDS, CompilerFlags.NESTED, CompilerFlags.NOFREE, CompilerFlags.OPTIMIZED, CompilerFlags.FUTURE_GENERATOR_STOP):
        self.assertTrue(bool(code.flags & f))
    code = Bytecode()
    flags = infer_flags(code)
    self.assertTrue(bool(flags & CompilerFlags.OPTIMIZED))
    self.assertTrue(bool(flags & CompilerFlags.NOFREE))
    code.append(ConcreteInstr('STORE_NAME', 1))
    flags = infer_flags(code)
    self.assertFalse(bool(flags & CompilerFlags.OPTIMIZED))
    self.assertTrue(bool(flags & CompilerFlags.NOFREE))
    code.append(ConcreteInstr('STORE_DEREF', 2))
    code.update_flags()
    self.assertFalse(bool(code.flags & CompilerFlags.OPTIMIZED))
    self.assertFalse(bool(code.flags & CompilerFlags.NOFREE))