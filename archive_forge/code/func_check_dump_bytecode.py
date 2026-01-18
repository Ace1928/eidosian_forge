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
def check_dump_bytecode(self, code, expected, lineno=None):
    with contextlib.redirect_stdout(io.StringIO()) as stderr:
        if lineno is not None:
            bytecode.dump_bytecode(code, lineno=True)
        else:
            bytecode.dump_bytecode(code)
        output = stderr.getvalue()
    self.assertEqual(output, expected)