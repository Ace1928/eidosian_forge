import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_backbone_loops(self):
    g = self.multiple_loops()
    self.assertEqual(sorted(g.backbone()), [0, 7, 60, 61, 68])
    g = self.infinite_loop1()
    self.assertEqual(sorted(g.backbone()), [0])
    g = self.infinite_loop2()
    self.assertEqual(sorted(g.backbone()), [0, 3])