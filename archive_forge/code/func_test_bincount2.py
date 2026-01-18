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
def test_bincount2(self):
    pyfunc = bincount2
    cfunc = jit(nopython=True)(pyfunc)
    for seq in self.bincount_sequences():
        w = [math.sqrt(x) - 2 for x in seq]
        for weights in (w, np.array(w), seq, np.array(seq)):
            expected = pyfunc(seq, weights)
            got = cfunc(seq, weights)
            self.assertPreciseEqual(expected, got)