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
def test_can_cast_timedelta(self):
    f = npdatetime_helpers.can_cast_timedelta_units
    for a, b in itertools.product(date_units, time_units):
        self.assertFalse(f(a, b), (a, b))
        self.assertFalse(f(b, a), (a, b))
    for unit in all_units:
        self.assertFalse(f(unit, ''))
        self.assertTrue(f('', unit))
    for unit in all_units + ('',):
        self.assertTrue(f(unit, unit))

    def check_units_group(group):
        for i, a in enumerate(group):
            for b in group[:i]:
                self.assertTrue(f(b, a))
                self.assertFalse(f(a, b))
    check_units_group(date_units)
    check_units_group(time_units)