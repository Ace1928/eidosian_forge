from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def lift_issue2561():
    np.empty(1)
    for i in range(10):
        for j in range(10):
            return 1
    return 2