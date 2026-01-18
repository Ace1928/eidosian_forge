import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_build_list_unpack_seq(self):
    for build_list in ('BUILD_TUPLE', 'BUILD_LIST'):
        code = Bytecode([Instr('LOAD_NAME', 'a'), Instr(build_list, 1), Instr('UNPACK_SEQUENCE', 1), Instr('STORE_NAME', 'x')])
        self.check(code, Instr('LOAD_NAME', 'a'), Instr('STORE_NAME', 'x'))
        code = Bytecode([Instr('LOAD_NAME', 'a'), Instr('LOAD_NAME', 'b'), Instr(build_list, 2), Instr('UNPACK_SEQUENCE', 2), Instr('STORE_NAME', 'x'), Instr('STORE_NAME', 'y')])
        self.check(code, Instr('LOAD_NAME', 'a'), Instr('LOAD_NAME', 'b'), Instr('ROT_TWO'), Instr('STORE_NAME', 'x'), Instr('STORE_NAME', 'y'))
        code = Bytecode([Instr('LOAD_NAME', 'a'), Instr('LOAD_NAME', 'b'), Instr('LOAD_NAME', 'c'), Instr(build_list, 3), Instr('UNPACK_SEQUENCE', 3), Instr('STORE_NAME', 'x'), Instr('STORE_NAME', 'y'), Instr('STORE_NAME', 'z')])
        self.check(code, Instr('LOAD_NAME', 'a'), Instr('LOAD_NAME', 'b'), Instr('LOAD_NAME', 'c'), Instr('ROT_THREE'), Instr('ROT_TWO'), Instr('STORE_NAME', 'x'), Instr('STORE_NAME', 'y'), Instr('STORE_NAME', 'z'))