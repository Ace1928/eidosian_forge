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
def test_datetime_timedelta_scaling(self):
    f = npdatetime_helpers.get_datetime_timedelta_conversion

    def check_error(dt_unit, td_unit):
        with self.assertRaises(RuntimeError):
            f(dt_unit, td_unit)
    for dt_unit, td_unit in itertools.product(time_units, date_units):
        check_error(dt_unit, td_unit)
    for dt_unit, td_unit in itertools.product(time_units, time_units):
        f(dt_unit, td_unit)
    for dt_unit, td_unit in itertools.product(date_units, time_units):
        f(dt_unit, td_unit)
    for dt_unit, td_unit in itertools.product(date_units, date_units):
        f(dt_unit, td_unit)
    for unit in all_units:
        self.assertEqual(f(unit, unit), (unit, 1, 1))
        self.assertEqual(f(unit, ''), (unit, 1, 1))
        self.assertEqual(f('', unit), ('', 1, 1))
    self.assertEqual(f('', ''), ('', 1, 1))
    self.assertEqual(f('Y', 'M'), ('M', 12, 1))
    self.assertEqual(f('M', 'Y'), ('M', 1, 12))
    self.assertEqual(f('W', 'D'), ('D', 7, 1))
    self.assertEqual(f('D', 'W'), ('D', 1, 7))
    self.assertEqual(f('W', 's'), ('s', 7 * 24 * 3600, 1))
    self.assertEqual(f('s', 'W'), ('s', 1, 7 * 24 * 3600))
    self.assertEqual(f('s', 'as'), ('as', 1000 ** 6, 1))
    self.assertEqual(f('as', 's'), ('as', 1, 1000 ** 6))
    self.assertEqual(f('Y', 'D'), ('D', 97 + 400 * 365, 400))
    self.assertEqual(f('Y', 'W'), ('W', 97 + 400 * 365, 400 * 7))
    self.assertEqual(f('M', 'D'), ('D', 97 + 400 * 365, 400 * 12))
    self.assertEqual(f('M', 'W'), ('W', 97 + 400 * 365, 400 * 12 * 7))
    self.assertEqual(f('Y', 's'), ('s', (97 + 400 * 365) * 24 * 3600, 400))
    self.assertEqual(f('M', 's'), ('s', (97 + 400 * 365) * 24 * 3600, 400 * 12))