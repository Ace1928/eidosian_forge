import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_build_map_from_local_vars(self):
    self.check(build_map_from_local_vars)