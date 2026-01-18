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
def diff_arrays(self):
    """
        Some test arrays for np.diff()
        """
    a = np.arange(12) ** 3
    yield a
    b = a.reshape((3, 4))
    yield b
    c = np.arange(24).reshape((3, 2, 4)) ** 3
    yield c