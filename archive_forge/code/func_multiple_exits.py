import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def multiple_exits(self):
    """
        A CFG with three loop exits, one of which is also a function
        exit point, and another function exit point:

            for x in a:
                if a:
                    return b
                elif b:
                    break
            return c
        """
    g = self.from_adj_list({0: [7], 7: [10, 36], 10: [19, 23], 19: [], 23: [29, 7], 29: [37], 36: [37], 37: []})
    g.set_entry_point(0)
    g.process()
    return g