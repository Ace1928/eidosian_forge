import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
@pytest.mark.skipif(not IS_PYPY, reason='Test is PyPy only (gh-9972)')
def test_int_from_infinite_longdouble___int__(self):
    x = np.longdouble(np.inf)
    assert_raises(OverflowError, x.__int__)
    with suppress_warnings() as sup:
        sup.record(np.ComplexWarning)
        x = np.clongdouble(np.inf)
        assert_raises(OverflowError, x.__int__)
        assert_equal(len(sup.log), 1)