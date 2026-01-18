import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.special import logsumexp
from scipy.stats import circstd
from ...data import from_dict, load_arviz_data
from ...stats.density_utils import histogram
from ...stats.stats_utils import (
from ...stats.stats_utils import logsumexp as _logsumexp
from ...stats.stats_utils import make_ufunc, not_valid, stats_variance_2d, wrap_xarray_ufunc
def test_circfunc():
    school = load_arviz_data('centered_eight').posterior['mu'].values
    a_a = _circfunc(school, 8, 4, skipna=False)
    assert np.allclose(a_a, _angle(school, 4, 8, np.pi))