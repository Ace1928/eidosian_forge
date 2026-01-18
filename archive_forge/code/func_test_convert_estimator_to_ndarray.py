from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize('array_namespace, converter', [('torch', lambda array: array.cpu().numpy()), ('numpy.array_api', lambda array: numpy.asarray(array)), ('cupy.array_api', lambda array: array._array.get())])
def test_convert_estimator_to_ndarray(array_namespace, converter):
    """Convert estimator attributes to ndarray."""
    xp = pytest.importorskip(array_namespace)
    X = xp.asarray([[1.3, 4.5]])
    est = SimpleEstimator().fit(X)
    new_est = _estimator_with_converted_arrays(est, converter)
    assert isinstance(new_est.X_, numpy.ndarray)