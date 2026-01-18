import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_combined_unary_bin_ops(self):
    code = Bytecode([Instr('LOAD_CONST', 1), Instr('LOAD_CONST', 3), Instr('BINARY_ADD'), Instr('LOAD_CONST', 7), Instr('BINARY_ADD'), Instr('STORE_NAME', 'x')])
    self.check(code, Instr('LOAD_CONST', 11), Instr('STORE_NAME', 'x'))
    code = Bytecode([Instr('LOAD_CONST', 5), Instr('UNARY_INVERT'), Instr('UNARY_INVERT'), Instr('STORE_NAME', 'x')])
    self.check(code, Instr('LOAD_CONST', 5), Instr('STORE_NAME', 'x'))
    code = Bytecode([Instr('LOAD_CONST', 0), Instr('LOAD_CONST', 'call'), Instr('BUILD_TUPLE', 2), Instr('LOAD_CONST', 1), Instr('LOAD_CONST', 'line'), Instr('BUILD_TUPLE', 2), Instr('LOAD_CONST', 3), Instr('UNARY_NEGATIVE'), Instr('LOAD_CONST', 'call'), Instr('BUILD_TUPLE', 2), Instr('BUILD_LIST', 3), Instr('STORE_NAME', 'events')])
    self.check(code, Instr('LOAD_CONST', (0, 'call')), Instr('LOAD_CONST', (1, 'line')), Instr('LOAD_CONST', (-3, 'call')), Instr('BUILD_LIST', 3), Instr('STORE_NAME', 'events'))
    code = Bytecode([Instr('LOAD_CONST', 1), Instr('BUILD_TUPLE', 1), Instr('LOAD_CONST', 0), Instr('BUILD_TUPLE', 1), Instr('LOAD_CONST', 8), Instr('BINARY_MULTIPLY'), Instr('BINARY_ADD'), Instr('STORE_NAME', 'x')])
    zeros = (0,) * 8
    result = (1,) + zeros
    self.check(code, Instr('LOAD_CONST', result), Instr('STORE_NAME', 'x'))