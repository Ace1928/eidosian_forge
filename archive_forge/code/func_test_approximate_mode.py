import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_approximate_mode():
    """Make sure sklearn.utils._approximate_mode returns valid
    results for cases where "class_counts * n_draws" is enough
    to overflow 32-bit signed integer.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20774
    """
    X = np.array([99000, 1000], dtype=np.int32)
    ret = _approximate_mode(class_counts=X, n_draws=25000, rng=0)
    assert_array_equal(ret, [24750, 250])