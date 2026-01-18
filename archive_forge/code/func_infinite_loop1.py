import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def infinite_loop1(self):
    """
        A CFG with a infinite loop and an alternate exit point:

            if c:
                return
            while True:
                if a:
                    ...
                else:
                    ...
        """
    g = self.from_adj_list({0: [10, 6], 6: [], 10: [13], 13: [26, 19], 19: [13], 26: [13]})
    g.set_entry_point(0)
    g.process()
    return g