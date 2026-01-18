import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase
def test_const_key_equal(self):
    neg_zero = -0.0
    pos_zero = +0.0
    self.assertNotEqual(Instr('LOAD_CONST', 0), Instr('LOAD_CONST', 0.0))
    self.assertNotEqual(Instr('LOAD_CONST', neg_zero), Instr('LOAD_CONST', pos_zero))
    self.assertNotEqual(Instr('LOAD_CONST', complex(neg_zero, 1.0)), Instr('LOAD_CONST', complex(pos_zero, 1.0)))
    self.assertNotEqual(Instr('LOAD_CONST', complex(1.0, neg_zero)), Instr('LOAD_CONST', complex(1.0, pos_zero)))
    self.assertNotEqual(Instr('LOAD_CONST', (0,)), Instr('LOAD_CONST', (0.0,)))
    nested_tuple1 = (0,)
    nested_tuple1 = (nested_tuple1,)
    nested_tuple2 = (0.0,)
    nested_tuple2 = (nested_tuple2,)
    self.assertNotEqual(Instr('LOAD_CONST', nested_tuple1), Instr('LOAD_CONST', nested_tuple2))
    self.assertNotEqual(Instr('LOAD_CONST', frozenset({0})), Instr('LOAD_CONST', frozenset({0.0})))