import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def check_getslice(self, obj):
    self._check_unary(getslice_usecase, obj, 1, len(obj) - 1)