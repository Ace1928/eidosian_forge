import pytest
import numpy as np
import scipy.stats
from ...stats.ecdf_utils import (
def test_compute_ecdf():
    """Test compute_ecdf function."""
    sample = np.array([1, 2, 3, 3, 4, 5])
    eval_points = np.arange(0, 7, 0.1)
    ecdf_expected = (sample[:, None] <= eval_points).mean(axis=0)
    assert np.allclose(compute_ecdf(sample, eval_points), ecdf_expected)
    assert np.allclose(compute_ecdf(sample / 2 + 10, eval_points / 2 + 10), ecdf_expected)