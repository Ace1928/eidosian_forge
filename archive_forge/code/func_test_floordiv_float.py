import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def test_floordiv_float(self):
    self.check_divmod_float(floordiv, [5.0, float('inf'), float('nan'), 2.0], ['divide by zero encountered', 'invalid value encountered'])