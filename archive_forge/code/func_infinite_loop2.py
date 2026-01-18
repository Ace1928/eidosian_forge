import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def infinite_loop2(self):
    """
        A CFG with no exit point at all:

            while True:
                if a:
                    ...
                else:
                    ...
        """
    g = self.from_adj_list({0: [3], 3: [16, 9], 9: [3], 16: [3]})
    g.set_entry_point(0)
    g.process()
    return g