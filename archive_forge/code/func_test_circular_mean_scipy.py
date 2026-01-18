from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
@pytest.mark.parametrize('mean', [0, np.pi, 4 * np.pi, -2 * np.pi, -10 * np.pi])
def test_circular_mean_scipy(mean):
    """Test our `_circular_mean()` function gives same result than Scipy version."""
    rvs = st.vonmises.rvs(loc=mean, kappa=1, size=1000)
    mean_az = _circular_mean(rvs)
    mean_sp = st.circmean(rvs, low=-np.pi, high=np.pi)
    np.testing.assert_almost_equal(mean_az, mean_sp)