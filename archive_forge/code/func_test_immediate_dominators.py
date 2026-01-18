import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_immediate_dominators(self):

    def check(graph, expected):
        idoms = graph.immediate_dominators()
        self.assertEqual(idoms, expected)
    check(self.loopless1(), {0: 0, 12: 0, 18: 0, 21: 0})
    check(self.loopless2(), {18: 99, 12: 99, 21: 99, 42: 21, 34: 21, 99: 99})
    check(self.loopless1_dead_nodes(), {0: 0, 12: 0, 18: 0, 21: 0})
    check(self.multiple_loops(), {0: 0, 7: 0, 10: 7, 13: 10, 20: 13, 23: 20, 32: 23, 44: 23, 56: 20, 57: 56, 60: 7, 61: 60, 68: 61, 71: 68, 80: 71, 87: 68, 88: 87})
    check(self.multiple_exits(), {0: 0, 7: 0, 10: 7, 19: 10, 23: 10, 29: 23, 36: 7, 37: 7})
    check(self.infinite_loop1(), {0: 0, 6: 0, 10: 0, 13: 10, 19: 13, 26: 13})
    check(self.infinite_loop2(), {0: 0, 3: 0, 9: 3, 16: 3})