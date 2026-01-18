import contextlib
import itertools
import re
import unittest
import warnings
import numpy as np
from numba import jit, vectorize, njit
from numba.np.numpy_support import numpy_version
from numba.core import types, config
from numba.core.errors import TypingError
from numba.tests.support import TestCase, tag, skip_parfors_unsupported
from numba.np import npdatetime_helpers, numpy_support
def test_homogeneous_div(self):
    div = self.jit(div_usecase)

    def check(a, b, expected):
        self.assertPreciseEqual(div(a, b), expected)
    check(TD(7), TD(3), 7.0 / 3.0)
    check(TD(7, 'us'), TD(3, 'ms'), 7.0 / 3000.0)
    check(TD(7, 'ms'), TD(3, 'us'), 7000.0 / 3.0)
    check(TD(7), TD(0), float('+inf'))
    check(TD(-7), TD(0), float('-inf'))
    check(TD(0), TD(0), float('nan'))
    check(TD('nat'), TD(3), float('nan'))
    check(TD(3), TD('nat'), float('nan'))
    check(TD('nat'), TD(0), float('nan'))
    with self.assertRaises((TypeError, TypingError)):
        div(TD(1, 'M'), TD(1, 'D'))