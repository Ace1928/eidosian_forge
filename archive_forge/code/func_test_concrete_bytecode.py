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
def test_concrete_bytecode(self):
    source = '\n            def func(test):\n                if test == 1:\n                    return 1\n                elif test == 2:\n                    return 2\n                return 3\n        '
    code = disassemble(source, function=True)
    code = code.to_concrete_bytecode()
    expected = f'\n  0    LOAD_FAST 0\n  2    LOAD_CONST 1\n  4    COMPARE_OP 2\n  6    POP_JUMP_IF_FALSE {(6 if OFFSET_AS_INSTRUCTION else 12)}\n  8    LOAD_CONST 1\n 10    RETURN_VALUE\n 12    LOAD_FAST 0\n 14    LOAD_CONST 2\n 16    COMPARE_OP 2\n 18    POP_JUMP_IF_FALSE {(12 if OFFSET_AS_INSTRUCTION else 24)}\n 20    LOAD_CONST 2\n 22    RETURN_VALUE\n 24    LOAD_CONST 3\n 26    RETURN_VALUE\n'.lstrip('\n')
    self.check_dump_bytecode(code, expected)
    expected = f'\nL.  2   0: LOAD_FAST 0\n        2: LOAD_CONST 1\n        4: COMPARE_OP 2\n        6: POP_JUMP_IF_FALSE {(6 if OFFSET_AS_INSTRUCTION else 12)}\nL.  3   8: LOAD_CONST 1\n       10: RETURN_VALUE\nL.  4  12: LOAD_FAST 0\n       14: LOAD_CONST 2\n       16: COMPARE_OP 2\n       18: POP_JUMP_IF_FALSE {(12 if OFFSET_AS_INSTRUCTION else 24)}\nL.  5  20: LOAD_CONST 2\n       22: RETURN_VALUE\nL.  6  24: LOAD_CONST 3\n       26: RETURN_VALUE\n'.lstrip('\n')
    self.check_dump_bytecode(code, expected, lineno=True)