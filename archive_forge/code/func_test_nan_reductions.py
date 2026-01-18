from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize('library', ['numpy', 'numpy.array_api', 'cupy', 'cupy.array_api', 'torch'])
@pytest.mark.parametrize('X,reduction,expected', [([1, 2, numpy.nan], _nanmin, 1), ([1, -2, -numpy.nan], _nanmin, -2), ([numpy.inf, numpy.inf], _nanmin, numpy.inf), ([[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]], partial(_nanmin, axis=0), [1.0, 2.0, 3.0]), ([[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]], partial(_nanmin, axis=1), [1.0, numpy.nan, 4.0]), ([1, 2, numpy.nan], _nanmax, 2), ([1, 2, numpy.nan], _nanmax, 2), ([-numpy.inf, -numpy.inf], _nanmax, -numpy.inf), ([[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]], partial(_nanmax, axis=0), [4.0, 5.0, 6.0]), ([[1, 2, 3], [numpy.nan, numpy.nan, numpy.nan], [4, 5, 6.0]], partial(_nanmax, axis=1), [3.0, numpy.nan, 6.0])])
def test_nan_reductions(library, X, reduction, expected):
    """Check NaN reductions like _nanmin and _nanmax"""
    xp = pytest.importorskip(library)
    if isinstance(expected, list):
        expected = xp.asarray(expected)
    with config_context(array_api_dispatch=True):
        result = reduction(xp.asarray(X))
    assert_allclose(result, expected)