import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_compare_op_unary_not(self):
    for op, not_op in ((Compare.IN, Compare.NOT_IN), (Compare.NOT_IN, Compare.IN), (Compare.IS, Compare.IS_NOT), (Compare.IS_NOT, Compare.IS)):
        code = Bytecode([Instr('LOAD_NAME', 'a'), Instr('LOAD_NAME', 'b'), Instr('COMPARE_OP', op), Instr('UNARY_NOT'), Instr('STORE_NAME', 'x')])
        self.check(code, Instr('LOAD_NAME', 'a'), Instr('LOAD_NAME', 'b'), Instr('COMPARE_OP', not_op), Instr('STORE_NAME', 'x'))
    label_instr5 = Label()
    code = Bytecode([Instr('LOAD_NAME', 'a'), Instr('JUMP_IF_FALSE_OR_POP', label_instr5), Instr('LOAD_NAME', 'b'), Instr('LOAD_CONST', True), Instr('COMPARE_OP', Compare.IS), label_instr5, Instr('UNARY_NOT'), Instr('STORE_NAME', 'x'), Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
    self.check_dont_optimize(code)