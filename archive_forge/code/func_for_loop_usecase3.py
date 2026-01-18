import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def for_loop_usecase3(x, y):
    result = 0
    lst = [x, y]
    for i in lst:
        result += i
    return result