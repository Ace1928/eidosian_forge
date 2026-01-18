import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_build_list_unpack_seq_const(self):
    code = Bytecode([Instr('LOAD_CONST', 3), Instr('LOAD_CONST', 4), Instr('LOAD_CONST', 5), Instr('BUILD_LIST', 3), Instr('UNPACK_SEQUENCE', 3), Instr('STORE_NAME', 'x'), Instr('STORE_NAME', 'y'), Instr('STORE_NAME', 'z')])
    self.check(code, Instr('LOAD_CONST', 5), Instr('LOAD_CONST', 4), Instr('LOAD_CONST', 3), Instr('STORE_NAME', 'x'), Instr('STORE_NAME', 'y'), Instr('STORE_NAME', 'z'))