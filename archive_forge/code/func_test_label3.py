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
def test_label3(self):
    """
        CPython generates useless EXTENDED_ARG 0 in some cases. We need to
        properly track them as otherwise we can end up with broken offset for
        jumps.
        """
    source = '\n            def func(x):\n                if x == 1:\n                    return x + 0\n                elif x == 2:\n                    return x + 1\n                elif x == 3:\n                    return x + 2\n                elif x == 4:\n                    return x + 3\n                elif x == 5:\n                    return x + 4\n                elif x == 6:\n                    return x + 5\n                elif x == 7:\n                    return x + 6\n                elif x == 8:\n                    return x + 7\n                elif x == 9:\n                    return x + 8\n                elif x == 10:\n                    return x + 9\n                elif x == 11:\n                    return x + 10\n                elif x == 12:\n                    return x + 11\n                elif x == 13:\n                    return x + 12\n                elif x == 14:\n                    return x + 13\n                elif x == 15:\n                    return x + 14\n                elif x == 16:\n                    return x + 15\n                elif x == 17:\n                    return x + 16\n                return -1\n        '
    code = get_code(source, function=True)
    bcode = Bytecode.from_code(code)
    concrete = bcode.to_concrete_bytecode()
    self.assertIsInstance(concrete, ConcreteBytecode)
    loc = {}
    exec(textwrap.dedent(source), loc)
    func = loc['func']
    func.__code__ = bcode.to_code()
    for i, x in enumerate(range(1, 18)):
        self.assertEqual(func(x), x + i)
    self.assertEqual(func(18), -1)
    self.assertEqual(ConcreteBytecode.from_code(code).to_code().co_code, code.co_code)