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
def make_jitclass_container(self):
    spec = {'data': types.List(dtype=types.List(types.float64[::1]))}
    JCContainer = jitclass(spec)(Container)
    return JCContainer