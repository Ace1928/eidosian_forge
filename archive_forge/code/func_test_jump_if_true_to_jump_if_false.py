import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_jump_if_true_to_jump_if_false(self):
    label_instr3 = Label()
    label_instr7 = Label()
    code = Bytecode([Instr('LOAD_NAME', 'x'), Instr('JUMP_IF_TRUE_OR_POP', label_instr3), Instr('LOAD_NAME', 'y'), label_instr3, Instr('POP_JUMP_IF_FALSE', label_instr7), Instr('LOAD_CONST', 1), Instr('STORE_NAME', 'z'), label_instr7, Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
    label_instr4 = Label()
    label_instr7 = Label()
    self.check(code, Instr('LOAD_NAME', 'x'), Instr('POP_JUMP_IF_TRUE', label_instr4), Instr('LOAD_NAME', 'y'), Instr('POP_JUMP_IF_FALSE', label_instr7), label_instr4, Instr('LOAD_CONST', 1), Instr('STORE_NAME', 'z'), label_instr7, Instr('LOAD_CONST', None), Instr('RETURN_VALUE'))