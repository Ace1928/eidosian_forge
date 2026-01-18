import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_dont_optimize(self):
    code = Bytecode([Instr('LOAD_CONST', 3), Instr('LOAD_CONST', 5), Instr('COMPARE_OP', Compare.LT), Instr('STORE_NAME', 'x'), Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
    self.check_dont_optimize(code)
    code = Bytecode([Instr('LOAD_CONST', (10, 20, 30)), Instr('LOAD_CONST', 1), Instr('LOAD_CONST', None), Instr('BUILD_SLICE', 2), Instr('BINARY_SUBSCR'), Instr('STORE_NAME', 'x')])
    self.check_dont_optimize(code)