from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def reject_npm1(x):
    a = np.empty(3, dtype=np.int32)
    for i in range(a.size):
        _ = object()
        a[i] = np.arange(i + 1)[i]
    return a