from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
def test_conditional_vect_numba_decorator():
    """Tests to see if Numba is used.

    Test can be distinguished from test_conditional_jit_decorator_no_numba
    by use of debugger or coverage tool
    """
    from ... import utils

    @utils.conditional_vect
    def func(a_a, b_b):
        return a_a + b_b
    value_one = np.random.randn(10)
    value_two = np.random.randn(10)
    assert np.allclose(func(value_one, value_two), value_one + value_two)