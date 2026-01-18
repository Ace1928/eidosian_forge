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
def test_modular_power(self):
    a = 5
    b = 4
    c = 10
    expected = pow(a, b, c)
    for t in (np.int32, np.float32, np.complex64):
        assert_raises(TypeError, operator.pow, t(a), b, c)
        assert_raises(TypeError, operator.pow, np.array(t(a)), b, c)