import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def loopless1_dead_nodes(self):
    """
        Same as loopless1(), but with added dead blocks (some of them
        in a loop).
        """
    g = self.from_adj_list({0: [18, 12], 12: [21], 18: [21], 21: [], 91: [12, 0], 92: [91, 93], 93: [92], 94: []})
    g.set_entry_point(0)
    g.process()
    return g