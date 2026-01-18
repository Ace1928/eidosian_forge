import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import sys
import textwrap
import types
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import get_code, TestCase
def test_extended_jump(self):
    NOP = bytes((opcode.opmap['NOP'],))

    class BigInstr(ConcreteInstr):

        def __init__(self, size):
            super().__init__('NOP')
            self._size = size

        def copy(self):
            return self

        def assemble(self):
            return NOP * self._size
    label = Label()
    nb_nop = 2 ** 16
    code = Bytecode([Instr('JUMP_ABSOLUTE', label), BigInstr(nb_nop), label, Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
    code_obj = code.to_code()
    if OFFSET_AS_INSTRUCTION:
        expected = b'\x90\x80q\x02' + NOP * nb_nop + b'd\x00S\x00'
    else:
        expected = b'\x90\x01\x90\x00q\x06' + NOP * nb_nop + b'd\x00S\x00'
    self.assertEqual(code_obj.co_code, expected)