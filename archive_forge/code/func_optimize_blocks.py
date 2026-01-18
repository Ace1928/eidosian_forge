import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import sys
import unittest
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Compare, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode import peephole_opt
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase, dump_bytecode
from unittest import mock
def optimize_blocks(self, code):
    if isinstance(code, Bytecode):
        code = ControlFlowGraph.from_bytecode(code)
    optimizer = peephole_opt.PeepholeOptimizer()
    optimizer.optimize_cfg(code)
    return code