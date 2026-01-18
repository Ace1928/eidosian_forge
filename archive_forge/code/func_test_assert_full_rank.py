from __future__ import print_function
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.util import (atleast_2d_column_default,
from patsy.desc import Term, INTERCEPT
from patsy.build import *
from patsy.categorical import C
from patsy.user_util import balanced, LookupFactor
from patsy.design_info import DesignMatrix, DesignInfo
def test_assert_full_rank():
    assert_full_rank(np.eye(10))
    assert_full_rank([[1, 0], [1, 0], [1, 0], [1, 1]])
    pytest.raises(AssertionError, assert_full_rank, [[1, 0], [2, 0]])
    pytest.raises(AssertionError, assert_full_rank, [[1, 2], [2, 4]])
    pytest.raises(AssertionError, assert_full_rank, [[1, 2, 3], [1, 10, 100]])
    pytest.raises(AssertionError, assert_full_rank, [[1, 2, 3], [1, 5, 6], [1, 6, 7]])