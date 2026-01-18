import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from scipy.stats import CensoredData
def test_invalid_constructor_args(self):
    with pytest.raises(ValueError, match='must be a one-dimensional'):
        CensoredData(uncensored=[[1, 2, 3]])
    with pytest.raises(ValueError, match='must be a one-dimensional'):
        CensoredData(left=[[1, 2, 3]])
    with pytest.raises(ValueError, match='must be a one-dimensional'):
        CensoredData(right=[[1, 2, 3]])
    with pytest.raises(ValueError, match='must be a two-dimensional'):
        CensoredData(interval=[[1, 2, 3]])
    with pytest.raises(ValueError, match='must not contain nan'):
        CensoredData(uncensored=[1, np.nan, 2])
    with pytest.raises(ValueError, match='must not contain nan'):
        CensoredData(left=[1, np.nan, 2])
    with pytest.raises(ValueError, match='must not contain nan'):
        CensoredData(right=[1, np.nan, 2])
    with pytest.raises(ValueError, match='must not contain nan'):
        CensoredData(interval=[[1, np.nan], [2, 3]])
    with pytest.raises(ValueError, match='both values must not be infinite'):
        CensoredData(interval=[[1, 3], [2, 9], [np.inf, np.inf]])
    with pytest.raises(ValueError, match='left value must not exceed the right'):
        CensoredData(interval=[[1, 0], [2, 2]])