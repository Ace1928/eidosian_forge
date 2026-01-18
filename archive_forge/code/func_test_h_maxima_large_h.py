import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
def test_h_maxima_large_h(self):
    """test that h-maxima works correctly for large h"""
    data = np.array([[10, 10, 10, 10, 10], [10, 13, 13, 13, 10], [10, 13, 14, 13, 10], [10, 13, 13, 13, 10], [10, 10, 10, 10, 10]], dtype=np.uint8)
    maxima = extrema.h_maxima(data, 5)
    assert np.sum(maxima) == 0
    data = np.array([[10, 10, 10, 10, 10], [10, 13, 13, 13, 10], [10, 13, 14, 13, 10], [10, 13, 13, 13, 10], [10, 10, 10, 10, 10]], dtype=np.float32)
    maxima = extrema.h_maxima(data, 5.0)
    assert np.sum(maxima) == 0