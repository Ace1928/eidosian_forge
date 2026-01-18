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
def test_combine_datetime_timedelta_units(self):
    f = npdatetime_helpers.combine_datetime_timedelta_units
    for unit in all_units:
        self.assertEqual(f(unit, unit), unit)
        self.assertEqual(f('', unit), unit)
        self.assertEqual(f(unit, ''), unit)
    self.assertEqual(f('', ''), '')
    for dt_unit, td_unit in itertools.product(time_units, date_units):
        self.assertIs(f(dt_unit, td_unit), None)
    for dt_unit, td_unit in itertools.product(date_units, time_units):
        self.assertEqual(f(dt_unit, td_unit), td_unit)