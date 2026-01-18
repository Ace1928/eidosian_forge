import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase
def test_const_key_not_equal(self):

    def check(value):
        self.assertEqual(Instr('LOAD_CONST', value), Instr('LOAD_CONST', value))

    def func():
        pass
    check(None)
    check(0)
    check(0.0)
    check(b'bytes')
    check('text')
    check(Ellipsis)
    check((1, 2, 3))
    check(frozenset({1, 2, 3}))
    check(func.__code__)
    check(object())