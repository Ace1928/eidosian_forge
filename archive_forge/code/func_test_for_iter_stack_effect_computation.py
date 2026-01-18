import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, FreeVar, Bytecode, SetLineno, ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, get_code
def test_for_iter_stack_effect_computation(self):
    with self.subTest():
        code = Bytecode()
        code.first_lineno = 1
        lab1 = Label()
        lab2 = Label()
        code.extend([lab1, Instr('FOR_ITER', lab2), Instr('STORE_FAST', 'i'), Instr('JUMP_ABSOLUTE', lab1), lab2])
        with self.assertRaises(RuntimeError):
            code.compute_stacksize(check_pre_and_post=False)