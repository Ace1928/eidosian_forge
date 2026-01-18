import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_dead_nodes(self):
    g = self.loopless1()
    self.assertEqual(len(g.dead_nodes()), 0)
    self.assertEqual(sorted(g.nodes()), [0, 12, 18, 21])
    g = self.loopless2()
    self.assertEqual(len(g.dead_nodes()), 0)
    self.assertEqual(sorted(g.nodes()), [12, 18, 21, 34, 42, 99])
    g = self.multiple_loops()
    self.assertEqual(len(g.dead_nodes()), 0)
    g = self.infinite_loop1()
    self.assertEqual(len(g.dead_nodes()), 0)
    g = self.multiple_exits()
    self.assertEqual(len(g.dead_nodes()), 0)
    g = self.loopless1_dead_nodes()
    self.assertEqual(sorted(g.dead_nodes()), [91, 92, 93, 94])
    self.assertEqual(sorted(g.nodes()), [0, 12, 18, 21])