from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def lift_gen1(x):
    a = np.empty(3)
    yield 0
    for i in range(a.size):
        a[i] = x
    yield np.sum(a)