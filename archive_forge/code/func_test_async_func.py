import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import unittest
from _pydevd_frame_eval.vendored.bytecode import ConcreteBytecode, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode.tests import get_code
def test_async_func(self):
    self.check('\n            async def func(arg, arg2):\n                pass\n        ', function=True)