import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import contextlib
import io
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Bytecode, BasicBlock, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import disassemble
def skip_test_version(self):
    import setup
    self.assertEqual(bytecode.__version__, setup.VERSION)