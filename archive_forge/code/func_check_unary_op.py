import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def check_unary_op(op, value, result):
    code = Bytecode([Instr('LOAD_CONST', value), Instr(op), Instr('STORE_NAME', 'x')])
    self.check(code, Instr('LOAD_CONST', result), Instr('STORE_NAME', 'x'))