import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_dominance_frontier(self):

    def check(graph, expected):
        df = graph.dominance_frontier()
        self.assertEqual(df, expected)
    check(self.loopless1(), {0: set(), 12: {21}, 18: {21}, 21: set()})
    check(self.loopless2(), {18: {21}, 12: {21}, 21: set(), 42: set(), 34: set(), 99: set()})
    check(self.loopless1_dead_nodes(), {0: set(), 12: {21}, 18: {21}, 21: set()})
    check(self.multiple_loops(), {0: set(), 7: {7}, 10: {7}, 13: {7}, 20: {20, 7}, 23: {20}, 32: {20}, 44: {20}, 56: {7}, 57: {7}, 60: set(), 61: set(), 68: {68}, 71: {68}, 80: set(), 87: set(), 88: set()})
    check(self.multiple_exits(), {0: set(), 7: {7}, 10: {37, 7}, 19: set(), 23: {37, 7}, 29: {37}, 36: {37}, 37: set()})
    check(self.infinite_loop1(), {0: set(), 6: set(), 10: set(), 13: {13}, 19: {13}, 26: {13}})
    check(self.infinite_loop2(), {0: set(), 3: {3}, 9: {3}, 16: {3}})