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
def test_inplace_floordiv_handling(self):
    a = np.array([1, 2], np.int64)
    b = np.array([1, 2], np.uint64)
    with pytest.raises(TypeError, match="Cannot cast ufunc 'floor_divide' output from"):
        a //= b