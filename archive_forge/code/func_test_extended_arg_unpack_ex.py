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
def test_extended_arg_unpack_ex(self):

    def test():
        p = [1, 2, 3, 4, 5, 6]
        q, r, *s, t = p
        return (q, r, s, t)
    cpython_stacksize = test.__code__.co_stacksize
    test.__code__ = ConcreteBytecode.from_code(test.__code__, extended_arg=True).to_code()
    self.assertEqual(test.__code__.co_stacksize, cpython_stacksize)
    self.assertEqual(test(), (1, 2, [3, 4, 5], 6))