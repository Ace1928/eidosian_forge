import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_eq_ne_structured_extra(self):
    dt = np.dtype('i4,i4')
    for m1 in (mvoid((1, 2), mask=(0, 0), dtype=dt), mvoid((1, 2), mask=(0, 1), dtype=dt), mvoid((1, 2), mask=(1, 0), dtype=dt), mvoid((1, 2), mask=(1, 1), dtype=dt)):
        ma1 = m1.view(MaskedArray)
        r1 = ma1.view('2i4')
        for m2 in (np.array((1, 1), dtype=dt), mvoid((1, 1), dtype=dt), mvoid((1, 0), mask=(0, 1), dtype=dt), mvoid((3, 2), mask=(0, 1), dtype=dt)):
            ma2 = m2.view(MaskedArray)
            r2 = ma2.view('2i4')
            eq_expected = (r1 == r2).all()
            assert_equal(m1 == m2, eq_expected)
            assert_equal(m2 == m1, eq_expected)
            assert_equal(ma1 == m2, eq_expected)
            assert_equal(m1 == ma2, eq_expected)
            assert_equal(ma1 == ma2, eq_expected)
            el_by_el = [m1[name] == m2[name] for name in dt.names]
            assert_equal(array(el_by_el, dtype=bool).all(), eq_expected)
            ne_expected = (r1 != r2).any()
            assert_equal(m1 != m2, ne_expected)
            assert_equal(m2 != m1, ne_expected)
            assert_equal(ma1 != m2, ne_expected)
            assert_equal(m1 != ma2, ne_expected)
            assert_equal(ma1 != ma2, ne_expected)
            el_by_el = [m1[name] != m2[name] for name in dt.names]
            assert_equal(array(el_by_el, dtype=bool).any(), ne_expected)