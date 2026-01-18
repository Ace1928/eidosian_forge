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
def test_get_jump_target(self):
    jump_abs = ConcreteInstr('JUMP_ABSOLUTE', 3)
    self.assertEqual(jump_abs.get_jump_target(100), 3)
    jump_forward = ConcreteInstr('JUMP_FORWARD', 5)
    self.assertEqual(jump_forward.get_jump_target(10), 16 if OFFSET_AS_INSTRUCTION else 17)