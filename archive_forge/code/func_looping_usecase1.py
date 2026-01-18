import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
def looping_usecase1(rec):
    a = rec('a')
    b = rec('b')
    c = rec('c')
    x = b
    for y in a:
        x = x + y
        rec.mark('--loop bottom--')
    rec.mark('--loop exit--')
    x = x + c
    return x