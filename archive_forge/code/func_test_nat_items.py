import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_nat_items(self):
    nadt_no_unit = np.datetime64('NaT')
    nadt_s = np.datetime64('NaT', 's')
    nadt_d = np.datetime64('NaT', 'ns')
    natd_no_unit = np.timedelta64('NaT')
    natd_s = np.timedelta64('NaT', 's')
    natd_d = np.timedelta64('NaT', 'ns')
    dts = [nadt_no_unit, nadt_s, nadt_d]
    tds = [natd_no_unit, natd_s, natd_d]
    for a, b in itertools.product(dts, dts):
        self._assert_func(a, b)
        self._assert_func([a], [b])
        self._test_not_equal([a], b)
    for a, b in itertools.product(tds, tds):
        self._assert_func(a, b)
        self._assert_func([a], [b])
        self._test_not_equal([a], b)
    for a, b in itertools.product(tds, dts):
        self._test_not_equal(a, b)
        self._test_not_equal(a, [b])
        self._test_not_equal([a], [b])
        self._test_not_equal([a], np.datetime64('2017-01-01', 's'))
        self._test_not_equal([b], np.datetime64('2017-01-01', 's'))
        self._test_not_equal([a], np.timedelta64(123, 's'))
        self._test_not_equal([b], np.timedelta64(123, 's'))