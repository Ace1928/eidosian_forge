import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal
from skimage import data, filters
from skimage._shared.utils import _supported_float_type
from skimage.filters.edges import _mask_filter_result
@pytest.mark.parametrize(('func', 'max_edge'), [(filters.prewitt, MAX_SOBEL_0), (filters.sobel, MAX_SOBEL_0), (filters.scharr, MAX_SOBEL_0), (filters.farid, MAX_FARID_0)])
def test_3d_edge_filters_single_axis(func, max_edge):
    blobs = data.binary_blobs(length=128, n_dim=3, rng=5)
    edges0 = func(blobs, axis=0)
    center = max_edge.shape[0] // 2
    if center == 2:
        rtol = 0.001
    else:
        rtol = 1e-07
    assert_allclose(np.max(edges0), func(max_edge, axis=0)[center, center, center], rtol=rtol)