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
def list_setslice3_arbitrary(n, n_src, start, stop, step):
    l = list(range(n))
    l[start:stop:step] = list(range(100, 100 + n_src))
    return l