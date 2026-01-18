from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
@pytest.mark.parametrize('mean', [[0, 0], [1, 1]])
@pytest.mark.parametrize('cov', [np.diag([1, 1]), np.diag([0.5, 0.5]), np.diag([0.25, 1]), np.array([[0.4, 0.2], [0.2, 0.8]])])
@pytest.mark.parametrize('contour_sigma', [np.array([1, 2, 3])])
def test_find_hdi_contours(mean, cov, contour_sigma):
    """Test `_find_hdi_contours()` against SciPy's multivariate normal distribution."""
    prob_dist = st.multivariate_normal(mean, cov)
    eigenvals, eigenvecs = np.linalg.eig(cov)
    eigenvecs = eigenvecs.T
    stdevs = np.sqrt(eigenvals)
    extremes = np.empty((4, 2))
    for i in range(4):
        extremes[i] = mean + (-1) ** i * 7 * stdevs[i // 2] * eigenvecs[i // 2]
    x_min, y_min = np.amin(extremes, axis=0)
    x_max, y_max = np.amax(extremes, axis=0)
    x = np.linspace(x_min, x_max, 256)
    y = np.linspace(y_min, y_max, 256)
    grid = np.dstack(np.meshgrid(x, y))
    density = prob_dist.pdf(grid)
    contour_sp = np.empty(contour_sigma.shape)
    for idx, sigma in enumerate(contour_sigma):
        contour_sp[idx] = prob_dist.pdf(mean + sigma * stdevs[0] * eigenvecs[0])
    hdi_probs = 1 - np.exp(-0.5 * contour_sigma ** 2)
    contour_az = _find_hdi_contours(density, hdi_probs)
    np.testing.assert_allclose(contour_sp, contour_az, rtol=0.01, atol=0.0001)