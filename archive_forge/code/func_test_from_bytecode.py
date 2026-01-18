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
def test_from_bytecode(self):
    bytecode = Bytecode()
    label = Label()
    bytecode.extend([Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label, lineno=1), Instr('LOAD_CONST', 5, lineno=2), Instr('STORE_NAME', 'x', lineno=2), Instr('JUMP_FORWARD', label, lineno=2), Instr('LOAD_CONST', 7, lineno=4), Instr('STORE_NAME', 'x', lineno=4), Label(), label, Label(), Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)])
    blocks = ControlFlowGraph.from_bytecode(bytecode)
    label2 = blocks[3]
    self.assertBlocksEqual(blocks, [Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label2, lineno=1)], [Instr('LOAD_CONST', 5, lineno=2), Instr('STORE_NAME', 'x', lineno=2), Instr('JUMP_FORWARD', label2, lineno=2)], [Instr('LOAD_CONST', 7, lineno=4), Instr('STORE_NAME', 'x', lineno=4)], [Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)])