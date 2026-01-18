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
def check_same(a, b):
    if isinstance(a, types.LiteralList) and isinstance(b, types.LiteralList):
        for i, j in zip(a.literal_value, b.literal_value):
            check_same(a.literal_value, b.literal_value)
    elif isinstance(a, list) and isinstance(b, list):
        for i, j in zip(a, b):
            check_same(i, j)
    elif isinstance(a, types.LiteralStrKeyDict) and isinstance(b, types.LiteralStrKeyDict):
        for (ki, vi), (kj, vj) in zip(a.literal_value.items(), b.literal_value.items()):
            check_same(ki, kj)
            check_same(vi, vj)
    elif isinstance(a, dict) and isinstance(b, dict):
        for (ki, vi), (kj, vj) in zip(a.items(), b.items()):
            check_same(ki, kj)
            check_same(vi, vj)
    else:
        self.assertEqual(a, b)