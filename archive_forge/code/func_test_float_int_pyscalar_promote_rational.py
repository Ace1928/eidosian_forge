import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
@pytest.mark.parametrize(['other', 'expected'], [(1, rational), (1.0, np.float64)])
@np._no_nep50_warning()
def test_float_int_pyscalar_promote_rational(self, weak_promotion, other, expected):
    if not weak_promotion and type(other) == float:
        with pytest.raises(TypeError, match='.* do not have a common DType'):
            np.result_type(other, rational)
    else:
        assert np.result_type(other, rational) == expected
    assert np.result_type(other, rational(1, 2)) == expected