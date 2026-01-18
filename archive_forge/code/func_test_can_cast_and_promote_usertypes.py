import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def test_can_cast_and_promote_usertypes(self):
    valid_types = ['int8', 'int16', 'int32', 'int64', 'bool']
    invalid_types = 'BHILQP' + 'FDG' + 'mM' + 'f' + 'V'
    rational_dt = np.dtype(rational)
    for numpy_dtype in valid_types:
        numpy_dtype = np.dtype(numpy_dtype)
        assert np.can_cast(numpy_dtype, rational_dt)
        assert np.promote_types(numpy_dtype, rational_dt) is rational_dt
    for numpy_dtype in invalid_types:
        numpy_dtype = np.dtype(numpy_dtype)
        assert not np.can_cast(numpy_dtype, rational_dt)
        with pytest.raises(TypeError):
            np.promote_types(numpy_dtype, rational_dt)
    double_dt = np.dtype('double')
    assert np.can_cast(rational_dt, double_dt)
    assert np.promote_types(double_dt, rational_dt) is double_dt