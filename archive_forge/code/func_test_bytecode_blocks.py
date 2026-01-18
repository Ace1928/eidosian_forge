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
def test_bytecode_blocks(self):
    source = '\n            def func(test):\n                if test == 1:\n                    return 1\n                elif test == 2:\n                    return 2\n                return 3\n        '
    code = disassemble(source, function=True)
    code = ControlFlowGraph.from_bytecode(code)
    enum_repr = '<Compare.EQ: 2>'
    expected = textwrap.dedent(f"\n            block1:\n                LOAD_FAST 'test'\n                LOAD_CONST 1\n                COMPARE_OP {enum_repr}\n                POP_JUMP_IF_FALSE <block3>\n                -> block2\n\n            block2:\n                LOAD_CONST 1\n                RETURN_VALUE\n\n            block3:\n                LOAD_FAST 'test'\n                LOAD_CONST 2\n                COMPARE_OP {enum_repr}\n                POP_JUMP_IF_FALSE <block5>\n                -> block4\n\n            block4:\n                LOAD_CONST 2\n                RETURN_VALUE\n\n            block5:\n                LOAD_CONST 3\n                RETURN_VALUE\n\n        ").lstrip()
    self.check_dump_bytecode(code, expected)
    expected = textwrap.dedent(f"\n            block1:\n                L.  2   0: LOAD_FAST 'test'\n                        1: LOAD_CONST 1\n                        2: COMPARE_OP {enum_repr}\n                        3: POP_JUMP_IF_FALSE <block3>\n                -> block2\n\n            block2:\n                L.  3   0: LOAD_CONST 1\n                        1: RETURN_VALUE\n\n            block3:\n                L.  4   0: LOAD_FAST 'test'\n                        1: LOAD_CONST 2\n                        2: COMPARE_OP {enum_repr}\n                        3: POP_JUMP_IF_FALSE <block5>\n                -> block4\n\n            block4:\n                L.  5   0: LOAD_CONST 2\n                        1: RETURN_VALUE\n\n            block5:\n                L.  6   0: LOAD_CONST 3\n                        1: RETURN_VALUE\n\n        ").lstrip()
    self.check_dump_bytecode(code, expected, lineno=True)