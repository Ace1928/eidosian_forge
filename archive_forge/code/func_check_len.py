import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def check_len(self, obj):
    self._check_unary(len_usecase, obj)