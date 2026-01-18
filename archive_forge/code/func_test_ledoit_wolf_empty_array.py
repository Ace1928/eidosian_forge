import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import (
from sklearn.covariance._shrunk_covariance import _ledoit_wolf
from sklearn.utils._testing import (
from .._shrunk_covariance import _oas
@pytest.mark.parametrize('ledoit_wolf_fitting_function', [LedoitWolf().fit, ledoit_wolf_shrinkage])
def test_ledoit_wolf_empty_array(ledoit_wolf_fitting_function):
    """Check that we validate X and raise proper error with 0-sample array."""
    X_empty = np.zeros((0, 2))
    with pytest.raises(ValueError, match='Found array with 0 sample'):
        ledoit_wolf_fitting_function(X_empty)