import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
@pytest.mark.parametrize('width', [3, 8, 20, 50])
@pytest.mark.parametrize('height', [3, 8, 20, 50])
def test_ellipse_crosses_approximation(width, height):
    fp_func = footprints.ellipse
    expected = fp_func(width, height, decomposition=None)
    footprint_sequence = fp_func(width, height, decomposition='crosses')
    approximate = footprints.footprint_from_sequence(footprint_sequence)
    assert approximate.shape == expected.shape
    error = np.sum(np.abs(expected.astype(int) - approximate.astype(int)))
    max_error = 0.05
    assert error / expected.size <= max_error