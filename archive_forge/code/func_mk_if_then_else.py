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
def mk_if_then_else(depth):
    instructions = []
    for i in range(depth):
        label_else = Label()
        instructions.extend([Instr('LOAD_FAST', 'x'), Instr('POP_JUMP_IF_FALSE', label_else), Instr('LOAD_GLOBAL', 'f{}'.format(i)), Instr('RETURN_VALUE'), label_else])
    instructions.extend([Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
    return instructions