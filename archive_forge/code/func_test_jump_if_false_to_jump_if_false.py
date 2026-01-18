import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def test_jump_if_false_to_jump_if_false(self):
    if sys.version_info < (3, 8):
        label_instr1 = Label()
        label_instr15 = Label()
        label_instr17 = Label()
        label_instr9 = Label()
        code = Bytecode([Instr('SETUP_LOOP', label_instr17), label_instr1, Instr('LOAD_NAME', 'n'), Instr('LOAD_CONST', 0), Instr('COMPARE_OP', Compare.GT), Instr('JUMP_IF_FALSE_OR_POP', label_instr9), Instr('LOAD_NAME', 'start'), Instr('LOAD_CONST', 3), Instr('COMPARE_OP', Compare.GT), label_instr9, Instr('POP_JUMP_IF_FALSE', label_instr15), Instr('LOAD_NAME', 'func'), Instr('CALL_FUNCTION', 0), Instr('POP_TOP'), Instr('JUMP_ABSOLUTE', label_instr1), label_instr15, Instr('POP_BLOCK'), label_instr17, Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
        label_instr1 = Label()
        label_instr14 = Label()
        label_instr16 = Label()
        self.check(code, Instr('SETUP_LOOP', label_instr16), label_instr1, Instr('LOAD_NAME', 'n'), Instr('LOAD_CONST', 0), Instr('COMPARE_OP', Compare.GT), Instr('POP_JUMP_IF_FALSE', label_instr14), Instr('LOAD_NAME', 'start'), Instr('LOAD_CONST', 3), Instr('COMPARE_OP', Compare.GT), Instr('POP_JUMP_IF_FALSE', label_instr14), Instr('LOAD_NAME', 'func'), Instr('CALL_FUNCTION', 0), Instr('POP_TOP'), Instr('JUMP_ABSOLUTE', label_instr1), label_instr14, Instr('POP_BLOCK'), label_instr16, Instr('LOAD_CONST', None), Instr('RETURN_VALUE'))
    else:
        label_instr1 = Label()
        label_instr15 = Label()
        label_instr9 = Label()
        code = Bytecode([label_instr1, Instr('LOAD_NAME', 'n'), Instr('LOAD_CONST', 0), Instr('COMPARE_OP', Compare.GT), Instr('JUMP_IF_FALSE_OR_POP', label_instr9), Instr('LOAD_NAME', 'start'), Instr('LOAD_CONST', 3), Instr('COMPARE_OP', Compare.GT), label_instr9, Instr('POP_JUMP_IF_FALSE', label_instr15), Instr('LOAD_NAME', 'func'), Instr('CALL_FUNCTION', 0), Instr('POP_TOP'), Instr('JUMP_ABSOLUTE', label_instr1), label_instr15, Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
        label_instr1 = Label()
        label_instr14 = Label()
        self.check(code, label_instr1, Instr('LOAD_NAME', 'n'), Instr('LOAD_CONST', 0), Instr('COMPARE_OP', Compare.GT), Instr('POP_JUMP_IF_FALSE', label_instr14), Instr('LOAD_NAME', 'start'), Instr('LOAD_CONST', 3), Instr('COMPARE_OP', Compare.GT), Instr('POP_JUMP_IF_FALSE', label_instr14), Instr('LOAD_NAME', 'func'), Instr('CALL_FUNCTION', 0), Instr('POP_TOP'), Instr('JUMP_ABSOLUTE', label_instr1), label_instr14, Instr('LOAD_CONST', None), Instr('RETURN_VALUE'))