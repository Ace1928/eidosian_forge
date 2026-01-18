from multiprocessing import Pool
from multiprocessing.pool import Pool as PWL
import re
import math
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal, assert_
import pytest
from pytest import raises as assert_raises
import hypothesis.extra.numpy as npst
from hypothesis import given, strategies, reproduce_failure  # noqa: F401
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_equal
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
def test_contains_nan_2d(self):
    data1 = np.array([[1, 2], [3, 4]])
    assert not _contains_nan(data1)[0]
    data2 = np.array([[1, 2], [3, np.nan]])
    assert _contains_nan(data2)[0]
    data3 = np.array([['1', 2], [3, np.nan]])
    assert not _contains_nan(data3)[0]
    data4 = np.array([['1', 2], [3, np.nan]], dtype='object')
    assert _contains_nan(data4)[0]