import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def arrs():
    a_0 = np.arange(10, 50)
    k_0 = 20
    yield (a_0, k_0)
    a_1 = np.arange(6)
    k_1 = 10
    yield (a_1, k_1)
    single_val_a = np.asarray([20])
    k_in = 20
    k_out = 13
    yield (single_val_a, k_in)
    yield (single_val_a, k_out)
    empty_arr = np.asarray([])
    yield (empty_arr, k_out)
    bool_arr = np.array([True, False])
    yield (bool_arr, True)
    yield (bool_arr, k_0)
    np.random.seed(2)
    float_arr = np.random.rand(10)
    np.random.seed(2)
    rand_k = np.random.rand()
    present_k = float_arr[0]
    yield (float_arr, rand_k)
    yield (float_arr, present_k)
    complx_arr = float_arr.view(np.complex128)
    yield (complx_arr, complx_arr[0])
    yield (complx_arr, rand_k)
    np.random.seed(2)
    uint_arr = np.random.randint(10, size=15, dtype=np.uint8)
    yield (uint_arr, 5)
    yield (uint_arr, 25)