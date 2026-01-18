import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from scipy.stats import CensoredData
def test_invalid_censored_args(self):
    with pytest.raises(ValueError, match='`low` must be a one-dimensional'):
        CensoredData.interval_censored(low=[[3]], high=[4, 5])
    with pytest.raises(ValueError, match='`high` must be a one-dimensional'):
        CensoredData.interval_censored(low=[3], high=[[4, 5]])
    with pytest.raises(ValueError, match='`low` must not contain'):
        CensoredData.interval_censored([1, 2, np.nan], [0, 1, 1])
    with pytest.raises(ValueError, match='must have the same length'):
        CensoredData.interval_censored([1, 2, 3], [0, 0, 1, 1])