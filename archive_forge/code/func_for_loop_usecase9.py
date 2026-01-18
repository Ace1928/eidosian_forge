import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def for_loop_usecase9(x, y):
    z = 0
    for i in range(x):
        x = 0
        for j in range(x):
            if j == x / 2:
                z += j
                break
        else:
            z += y
    return z