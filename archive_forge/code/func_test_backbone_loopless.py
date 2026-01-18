import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_backbone_loopless(self):
    for g in [self.loopless1(), self.loopless1_dead_nodes()]:
        self.assertEqual(sorted(g.backbone()), [0, 21])
    g = self.loopless2()
    self.assertEqual(sorted(g.backbone()), [21, 99])