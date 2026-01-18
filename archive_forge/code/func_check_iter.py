import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def check_iter(self, obj):
    self._check_unary(iter_usecase, obj)