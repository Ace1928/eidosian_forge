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
def test_label_at_the_end(self):
    label = Label()
    code = Bytecode([Instr('LOAD_NAME', 'x'), Instr('UNARY_NOT'), Instr('POP_JUMP_IF_FALSE', label), Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'y'), label])
    cfg = ControlFlowGraph.from_bytecode(code)
    self.assertBlocksEqual(cfg, [Instr('LOAD_NAME', 'x'), Instr('UNARY_NOT'), Instr('POP_JUMP_IF_FALSE', cfg[2])], [Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'y')], [])