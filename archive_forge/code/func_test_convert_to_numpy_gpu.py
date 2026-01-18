from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize('library', ['cupy', 'torch', 'cupy.array_api'])
def test_convert_to_numpy_gpu(library):
    """Check convert_to_numpy for GPU backed libraries."""
    xp = pytest.importorskip(library)
    if library == 'torch':
        if not xp.backends.cuda.is_built():
            pytest.skip('test requires cuda')
        X_gpu = xp.asarray([1.0, 2.0, 3.0], device='cuda')
    else:
        X_gpu = xp.asarray([1.0, 2.0, 3.0])
    X_cpu = _convert_to_numpy(X_gpu, xp=xp)
    expected_output = numpy.asarray([1.0, 2.0, 3.0])
    assert_allclose(X_cpu, expected_output)