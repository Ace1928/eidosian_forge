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
def list_contains(n):
    l = list(range(n))
    return (0 in l, 1 in l, n - 1 in l, n in l, 0 not in l, 1 not in l, n - 1 not in l, n not in l)