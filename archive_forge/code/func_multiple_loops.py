import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def multiple_loops(self):
    """
        A CFG with multiple nested loops:

            for y in b:
                for x in a:
                    # This loop has two back edges
                    if b:
                        continue
                    else:
                        continue
            for z in c:
                if z:
                    return ...
        """
    g = self.from_adj_list({0: [7], 7: [10, 60], 10: [13], 13: [20], 20: [56, 23], 23: [32, 44], 32: [20], 44: [20], 56: [57], 57: [7], 60: [61], 61: [68], 68: [87, 71], 71: [80, 68], 80: [], 87: [88], 88: []})
    g.set_entry_point(0)
    g.process()
    return g