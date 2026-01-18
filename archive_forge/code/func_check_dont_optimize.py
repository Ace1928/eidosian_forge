import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def check_dont_optimize(self, code):
    code = ControlFlowGraph.from_bytecode(code)
    noopt = code.to_bytecode()
    optim = self.optimize_blocks(code)
    optim = optim.to_bytecode()
    self.assertEqual(optim, noopt)