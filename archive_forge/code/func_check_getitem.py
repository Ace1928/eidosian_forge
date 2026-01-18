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
def check_getitem(self, code):
    for block_index, block in enumerate(code):
        self.assertIs(code[block_index], block)
        self.assertIs(code[block], block)
        self.assertEqual(code.get_block_index(block), block_index)