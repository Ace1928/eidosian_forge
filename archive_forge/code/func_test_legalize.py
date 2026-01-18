import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
def test_legalize(self):
    code = Bytecode()
    code.first_lineno = 3
    code.extend([Instr('LOAD_CONST', 7), Instr('STORE_NAME', 'x'), Instr('LOAD_CONST', 8, lineno=4), Instr('STORE_NAME', 'y'), Label(), SetLineno(5), Instr('LOAD_CONST', 9, lineno=6), Instr('STORE_NAME', 'z')])
    code.legalize()
    self.assertListEqual(code, [Instr('LOAD_CONST', 7, lineno=3), Instr('STORE_NAME', 'x', lineno=3), Instr('LOAD_CONST', 8, lineno=4), Instr('STORE_NAME', 'y', lineno=4), Label(), Instr('LOAD_CONST', 9, lineno=5), Instr('STORE_NAME', 'z', lineno=5)])