import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
@unittest.skipIf(sys.version_info < (3, 6), 'Inexistent opcode')
def test_negative_size_build_const_map(self):
    code = Bytecode()
    code.first_lineno = 1
    code.extend([Instr('LOAD_CONST', ('a',)), Instr('BUILD_CONST_KEY_MAP', 1)])
    with self.assertRaises(RuntimeError):
        code.compute_stacksize()