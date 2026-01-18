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
def bool_list_usecase():
    l = [False]
    l[0] = True
    x = False
    for v in l:
        x = x ^ v
    return (l, x)