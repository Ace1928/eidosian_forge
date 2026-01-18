import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
@pytest.mark.parametrize('as_sequence', [tuple, None])
def test_mirror_footprint(as_sequence):
    footprint = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], np.uint8)
    expected_res = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    if as_sequence is not None:
        footprint = as_sequence([(footprint, 2), (footprint.T, 3)])
        expected_res = as_sequence([(expected_res, 2), (expected_res.T, 3)])
    actual_res = footprints.mirror_footprint(footprint)
    assert type(expected_res) is type(actual_res)
    assert_equal(expected_res, actual_res)