import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
@pytest.mark.parametrize('function, args, supports_sequence_decomposition', [(footprints.disk, (3,), True), (footprints.ball, (3,), True), (footprints.square, (3,), True), (footprints.cube, (3,), True), (footprints.diamond, (3,), True), (footprints.octahedron, (3,), True), (footprints.rectangle, (3, 4), True), (footprints.ellipse, (3, 4), False), (footprints.octagon, (3, 4), True), (footprints.star, (3,), False)])
@pytest.mark.parametrize('dtype', [np.uint8, np.float64])
def test_footprint_dtype(function, args, supports_sequence_decomposition, dtype):
    footprint = function(*args, dtype=dtype)
    assert footprint.dtype == dtype
    if supports_sequence_decomposition:
        sequence = function(*args, dtype=dtype, decomposition='sequence')
        assert all([fp_tuple[0].dtype == dtype for fp_tuple in sequence])