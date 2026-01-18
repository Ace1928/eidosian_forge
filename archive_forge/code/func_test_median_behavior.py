import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage
from skimage.filters import median, rank
from skimage._shared.testing import assert_stacklevel
@pytest.mark.parametrize('behavior, func, params', [('ndimage', ndimage.median_filter, {'size': (3, 3)}), ('rank', rank.median, {'footprint': np.ones((3, 3), dtype=np.uint8)})])
def test_median_behavior(image, behavior, func, params):
    assert_allclose(median(image, behavior=behavior), func(image, **params))