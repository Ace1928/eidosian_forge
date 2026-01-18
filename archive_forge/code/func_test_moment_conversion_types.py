import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_, assert_equal
from statsmodels.stats import moment_helpers
from statsmodels.stats.moment_helpers import (cov2corr, mvsk2mc, mc2mvsk,
@pytest.mark.parametrize('func_name', ['cum2mc', 'cum2mc', 'mc2cum', 'mc2mnc', 'mc2mvsk', 'mnc2cum', 'mnc2mc', 'mnc2mc', 'mvsk2mc', 'mvsk2mnc'])
def test_moment_conversion_types(func_name):
    func = getattr(moment_helpers, func_name)
    assert isinstance(func([1.0, 1, 0, 3]), list) or isinstance(func(np.array([1.0, 1, 0, 3])), (tuple, np.ndarray))
    assert isinstance(func(np.array([1.0, 1, 0, 3])), list) or isinstance(func(np.array([1.0, 1, 0, 3])), (tuple, np.ndarray))
    assert isinstance(func(tuple([1.0, 1, 0, 3])), list) or isinstance(func(np.array([1.0, 1, 0, 3])), (tuple, np.ndarray))