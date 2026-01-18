from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_gaussian_kde_monkeypatch():
    """Ugly, but people may rely on this.  See scipy pull request 123,
    specifically the linked ML thread "Width of the Gaussian in stats.kde".
    If it is necessary to break this later on, that is to be discussed on ML.
    """
    x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
    xs = np.linspace(-10, 10, num=50)
    kde = stats.gaussian_kde(x1)
    kde.covariance_factor = kde.silverman_factor
    kde._compute_covariance()
    y1 = kde(xs)
    kde2 = stats.gaussian_kde(x1, bw_method='silverman')
    y2 = kde2(xs)
    assert_array_almost_equal_nulp(y1, y2, nulp=10)