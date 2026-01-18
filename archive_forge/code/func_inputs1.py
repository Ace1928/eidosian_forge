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
def inputs1():
    yield np.array([1, 1 + 2 * np.pi])
    phase = np.linspace(0, np.pi, num=5)
    phase[3:] += np.pi
    yield phase
    yield np.arange(16).reshape((4, 4))
    yield np.arange(160, step=10).reshape((4, 4))
    yield np.arange(240, step=10).reshape((2, 3, 4))