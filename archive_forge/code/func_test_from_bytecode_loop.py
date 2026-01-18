import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import io
import sys
import unittest
import contextlib
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import disassemble as _disassemble, TestCase
def test_from_bytecode_loop(self):
    if sys.version_info < (3, 8):
        label_loop_start = Label()
        label_loop_exit = Label()
        label_loop_end = Label()
        code = Bytecode()
        code.extend((Instr('SETUP_LOOP', label_loop_end, lineno=1), Instr('LOAD_CONST', (1, 2, 3), lineno=1), Instr('GET_ITER', lineno=1), label_loop_start, Instr('FOR_ITER', label_loop_exit, lineno=1), Instr('STORE_NAME', 'x', lineno=1), Instr('LOAD_NAME', 'x', lineno=2), Instr('LOAD_CONST', 2, lineno=2), Instr('COMPARE_OP', Compare.EQ, lineno=2), Instr('POP_JUMP_IF_FALSE', label_loop_start, lineno=2), Instr('BREAK_LOOP', lineno=3), Instr('JUMP_ABSOLUTE', label_loop_start, lineno=4), Instr('JUMP_ABSOLUTE', label_loop_start, lineno=4), label_loop_exit, Instr('POP_BLOCK', lineno=4), label_loop_end, Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)))
        blocks = ControlFlowGraph.from_bytecode(code)
        expected = [[Instr('SETUP_LOOP', blocks[8], lineno=1)], [Instr('LOAD_CONST', (1, 2, 3), lineno=1), Instr('GET_ITER', lineno=1)], [Instr('FOR_ITER', blocks[7], lineno=1)], [Instr('STORE_NAME', 'x', lineno=1), Instr('LOAD_NAME', 'x', lineno=2), Instr('LOAD_CONST', 2, lineno=2), Instr('COMPARE_OP', Compare.EQ, lineno=2), Instr('POP_JUMP_IF_FALSE', blocks[2], lineno=2)], [Instr('BREAK_LOOP', lineno=3)], [Instr('JUMP_ABSOLUTE', blocks[2], lineno=4)], [Instr('JUMP_ABSOLUTE', blocks[2], lineno=4)], [Instr('POP_BLOCK', lineno=4)], [Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)]]
        self.assertBlocksEqual(blocks, *expected)
    else:
        label_loop_start = Label()
        label_loop_exit = Label()
        code = Bytecode()
        code.extend((Instr('LOAD_CONST', (1, 2, 3), lineno=1), Instr('GET_ITER', lineno=1), label_loop_start, Instr('FOR_ITER', label_loop_exit, lineno=1), Instr('STORE_NAME', 'x', lineno=1), Instr('LOAD_NAME', 'x', lineno=2), Instr('LOAD_CONST', 2, lineno=2), Instr('COMPARE_OP', Compare.EQ, lineno=2), Instr('POP_JUMP_IF_FALSE', label_loop_start, lineno=2), Instr('JUMP_ABSOLUTE', label_loop_exit, lineno=3), Instr('JUMP_ABSOLUTE', label_loop_start, lineno=4), Instr('JUMP_ABSOLUTE', label_loop_start, lineno=4), label_loop_exit, Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)))
        blocks = ControlFlowGraph.from_bytecode(code)
        expected = [[Instr('LOAD_CONST', (1, 2, 3), lineno=1), Instr('GET_ITER', lineno=1)], [Instr('FOR_ITER', blocks[6], lineno=1)], [Instr('STORE_NAME', 'x', lineno=1), Instr('LOAD_NAME', 'x', lineno=2), Instr('LOAD_CONST', 2, lineno=2), Instr('COMPARE_OP', Compare.EQ, lineno=2), Instr('POP_JUMP_IF_FALSE', blocks[1], lineno=2)], [Instr('JUMP_ABSOLUTE', blocks[6], lineno=3)], [Instr('JUMP_ABSOLUTE', blocks[1], lineno=4)], [Instr('JUMP_ABSOLUTE', blocks[1], lineno=4)], [Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)]]
        self.assertBlocksEqual(blocks, *expected)