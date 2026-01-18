import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
def test_negative_size_build(self):
    opnames = ('BUILD_TUPLE', 'BUILD_LIST', 'BUILD_SET')
    if sys.version_info >= (3, 6):
        opnames = (*opnames, 'BUILD_STRING')
    for opname in opnames:
        with self.subTest():
            code = Bytecode()
            code.first_lineno = 1
            code.extend([Instr(opname, 1)])
            with self.assertRaises(RuntimeError):
                code.compute_stacksize()