from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
import pytest
from numpy.testing import assert_
from numpy.testing import assert_almost_equal
from statsmodels.base.optimizer import (
def dummy_bounds_constraint_func(x):
    return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2