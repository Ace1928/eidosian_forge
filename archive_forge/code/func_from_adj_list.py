import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def from_adj_list(self, d, entry_point=0):
    """
        Build a CFGraph class from a dict of adjacency lists.
        """
    g = CFGraph()
    for node in d:
        g.add_node(node)
    for node, dests in d.items():
        for dest in dests:
            g.add_edge(node, dest)
    return g