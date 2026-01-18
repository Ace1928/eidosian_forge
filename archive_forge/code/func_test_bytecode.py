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
def test_bytecode(self):
    source = '\n            def func(test):\n                if test == 1:\n                    return 1\n                elif test == 2:\n                    return 2\n                return 3\n        '
    code = disassemble(source, function=True)
    enum_repr = '<Compare.EQ: 2>'
    expected = f"\n    LOAD_FAST 'test'\n    LOAD_CONST 1\n    COMPARE_OP {enum_repr}\n    POP_JUMP_IF_FALSE <label_instr6>\n    LOAD_CONST 1\n    RETURN_VALUE\n\nlabel_instr6:\n    LOAD_FAST 'test'\n    LOAD_CONST 2\n    COMPARE_OP {enum_repr}\n    POP_JUMP_IF_FALSE <label_instr13>\n    LOAD_CONST 2\n    RETURN_VALUE\n\nlabel_instr13:\n    LOAD_CONST 3\n    RETURN_VALUE\n\n        "[1:].rstrip(' ')
    self.check_dump_bytecode(code, expected)
    expected = f"\n    L.  2   0: LOAD_FAST 'test'\n            1: LOAD_CONST 1\n            2: COMPARE_OP {enum_repr}\n            3: POP_JUMP_IF_FALSE <label_instr6>\n    L.  3   4: LOAD_CONST 1\n            5: RETURN_VALUE\n\nlabel_instr6:\n    L.  4   7: LOAD_FAST 'test'\n            8: LOAD_CONST 2\n            9: COMPARE_OP {enum_repr}\n           10: POP_JUMP_IF_FALSE <label_instr13>\n    L.  5  11: LOAD_CONST 2\n           12: RETURN_VALUE\n\nlabel_instr13:\n    L.  6  14: LOAD_CONST 3\n           15: RETURN_VALUE\n\n        "[1:].rstrip(' ')
    self.check_dump_bytecode(code, expected, lineno=True)