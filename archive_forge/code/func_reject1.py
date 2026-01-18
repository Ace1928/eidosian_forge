from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def reject1(x):
    a = np.arange(4)
    for i in range(a.shape[0]):
        return a
    return a