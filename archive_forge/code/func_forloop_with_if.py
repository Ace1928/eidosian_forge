from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
@jit(void(int32, double[:]), forceobj=True)
def forloop_with_if(u, a):
    if u == 0:
        for i in range(a.shape[0]):
            a[i] = a[i] * 2.0
    else:
        for i in range(a.shape[0]):
            a[i] = a[i] + 1.0