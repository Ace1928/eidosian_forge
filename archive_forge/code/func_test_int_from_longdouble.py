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
def test_int_from_longdouble(self):
    x = np.longdouble(1.5)
    assert_equal(int(x), 1)
    x = np.longdouble(-10.5)
    assert_equal(int(x), -10)