import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
def test_allow_borders(self):
    """Test maxima detection at the image border."""
    result_with_boder = extrema.local_maxima(self.image, connectivity=1, allow_borders=True)
    assert result_with_boder.dtype == bool
    assert_equal(result_with_boder, self.expected_cross)
    expected_without_border = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
    result_without_border = extrema.local_maxima(self.image, connectivity=1, allow_borders=False)
    assert result_with_boder.dtype == bool
    assert_equal(result_without_border, expected_without_border)