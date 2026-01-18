import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_if_else_nested(self):

    def foo():
        if SET_BLOCK_A0:
            SET_BLOCK_A1
            if SET_BLOCK_B0:
                SET_BLOCK_B1
                a = 0
            else:
                if SET_BLOCK_C0:
                    SET_BLOCK_C1
                    a = 1
                else:
                    SET_BLOCK_C2
                    a = 2
                SET_BLOCK_D
            SET_BLOCK_E
        SET_BLOCK_F
        return a
    cfa, blkpts = self.get_cfa_and_namedblocks(foo)
    idoms = cfa.graph.immediate_dominators()
    self.assertEqual(blkpts['A0'], idoms[blkpts['A1']])
    self.assertEqual(blkpts['A1'], idoms[blkpts['B1']])
    self.assertEqual(blkpts['A1'], idoms[blkpts['C0']])
    self.assertEqual(blkpts['C0'], idoms[blkpts['D']])
    self.assertEqual(blkpts['A1'], idoms[blkpts['E']])
    self.assertEqual(blkpts['A0'], idoms[blkpts['F']])
    domfront = cfa.graph.dominance_frontier()
    self.assertFalse(domfront[blkpts['A0']])
    self.assertFalse(domfront[blkpts['F']])
    self.assertEqual({blkpts['E']}, domfront[blkpts['B1']])
    self.assertEqual({blkpts['D']}, domfront[blkpts['C1']])
    self.assertEqual({blkpts['E']}, domfront[blkpts['D']])
    self.assertEqual({blkpts['F']}, domfront[blkpts['E']])