import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import io
import sys
import unittest
import contextlib
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import disassemble as _disassemble, TestCase
def sample_code(self):
    code = disassemble('x = 1', remove_last_return_none=True)
    self.assertBlocksEqual(code, [Instr('LOAD_CONST', 1, lineno=1), Instr('STORE_NAME', 'x', lineno=1)])
    return code