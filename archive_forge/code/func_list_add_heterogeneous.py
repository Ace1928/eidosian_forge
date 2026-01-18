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
def list_add_heterogeneous():
    a = [1]
    b = [2.0]
    c = a + b
    d = b + a
    a.append(3)
    b.append(4.0)
    return (a, b, c, d)