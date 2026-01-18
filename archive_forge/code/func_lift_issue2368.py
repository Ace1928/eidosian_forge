from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def lift_issue2368(a, b):
    s = 0
    for e in a:
        s += e
    h = b.__hash__()
    return (s, h)