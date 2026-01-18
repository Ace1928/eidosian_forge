import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase
def test_code_object_containing_mutable_data(self):
    from _pydevd_frame_eval.vendored.bytecode import Bytecode, Instr
    from types import CodeType

    def f():

        def g():
            return 'value'
        return g
    f_code = Bytecode.from_code(f.__code__)
    instr_load_code = None
    mutable_datum = [4, 2]
    for each in f_code:
        if isinstance(each, Instr) and each.name == 'LOAD_CONST' and isinstance(each.arg, CodeType):
            instr_load_code = each
            break
    self.assertIsNotNone(instr_load_code)
    g_code = Bytecode.from_code(instr_load_code.arg)
    g_code[0].arg = mutable_datum
    instr_load_code.arg = g_code.to_code()
    f.__code__ = f_code.to_code()
    self.assertIs(f()(), mutable_datum)