from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
def list_setslice2(n, n_source, start, stop):
    l = list(range(n))
    v = list(range(100, 100 + n_source))
    l[start:stop] = v
    return l