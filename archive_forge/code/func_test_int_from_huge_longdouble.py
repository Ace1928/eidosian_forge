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
@pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble), reason='long double is same as double')
@pytest.mark.skipif(platform.machine().startswith('ppc'), reason='IBM double double')
def test_int_from_huge_longdouble(self):
    exp = np.finfo(np.double).maxexp - 1
    huge_ld = 2 * 1234 * np.longdouble(2) ** exp
    huge_i = 2 * 1234 * 2 ** exp
    assert_(huge_ld != np.inf)
    assert_equal(int(huge_ld), huge_i)