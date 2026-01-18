import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
def basic_array_access(a):
    return a[10]