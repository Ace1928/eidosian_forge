import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def loopless1(self):
    """
        A simple CFG corresponding to the following code structure:

            c = (... if ... else ...) + ...
            return b + c
        """
    g = self.from_adj_list({0: [18, 12], 12: [21], 18: [21], 21: []})
    g.set_entry_point(0)
    g.process()
    return g