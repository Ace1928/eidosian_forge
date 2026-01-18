import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
@pytest.mark.parametrize(('func', 'max_edge'), [(filters.prewitt, MAX_SOBEL_ND), (filters.sobel, MAX_SOBEL_ND), (filters.scharr, MAX_SCHARR_ND), (filters.farid, MAX_FARID_ND)])
def test_3d_edge_filters(func, max_edge):
    blobs = data.binary_blobs(length=128, n_dim=3, rng=5)
    edges = func(blobs)
    center = max_edge.shape[0] // 2
    if center == 2:
        rtol = 0.001
    else:
        rtol = 1e-07
    assert_allclose(np.max(edges), func(max_edge)[center, center, center], rtol=rtol)