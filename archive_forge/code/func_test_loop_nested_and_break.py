import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def test_loop_nested_and_break(self):

    def foo(n):
        SET_BLOCK_A
        while SET_BLOCK_B0:
            SET_BLOCK_B1
            while SET_BLOCK_C0:
                SET_BLOCK_C1
                if SET_BLOCK_D0:
                    SET_BLOCK_D1
                    break
                elif n:
                    SET_BLOCK_D2
                SET_BLOCK_E
            SET_BLOCK_F
        SET_BLOCK_G
    cfa, blkpts = self.get_cfa_and_namedblocks(foo)
    idoms = cfa.graph.immediate_dominators()
    self.assertEqual(blkpts['D0'], blkpts['C1'])
    if PYVERSION < (3, 10):
        self.assertEqual(blkpts['C0'], idoms[blkpts['C1']])
    domfront = cfa.graph.dominance_frontier()
    self.assertFalse(domfront[blkpts['A']])
    self.assertFalse(domfront[blkpts['G']])
    if PYVERSION < (3, 10):
        self.assertEqual({blkpts['B0']}, domfront[blkpts['B1']])
    if PYVERSION < (3, 10):
        self.assertEqual({blkpts['C0'], blkpts['F']}, domfront[blkpts['C1']])
    self.assertEqual({blkpts['F']}, domfront[blkpts['D1']])
    self.assertEqual({blkpts['E']}, domfront[blkpts['D2']])
    if PYVERSION < (3, 10):
        self.assertEqual({blkpts['C0']}, domfront[blkpts['E']])
        self.assertEqual({blkpts['B0']}, domfront[blkpts['F']])
        self.assertEqual({blkpts['B0']}, domfront[blkpts['B0']])