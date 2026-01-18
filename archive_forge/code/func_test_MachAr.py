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
@unittest.skipIf(numpy_version >= (1, 24), 'NumPy < 1.24 required')
def test_MachAr(self):
    attrs = ('ibeta', 'it', 'machep', 'eps', 'negep', 'epsneg', 'iexp', 'minexp', 'xmin', 'maxexp', 'xmax', 'irnd', 'ngrd', 'epsilon', 'tiny', 'huge', 'precision', 'resolution')
    self.check(machar, attrs)