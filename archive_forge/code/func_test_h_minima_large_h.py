import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
def test_h_minima_large_h(self):
    """test that h-minima works correctly for large h"""
    data = np.array([[14, 14, 14, 14, 14], [14, 11, 11, 11, 14], [14, 11, 10, 11, 14], [14, 11, 11, 11, 14], [14, 14, 14, 14, 14]], dtype=np.uint8)
    maxima = extrema.h_minima(data, 5)
    assert np.sum(maxima) == 0
    data = np.array([[14, 14, 14, 14, 14], [14, 11, 11, 11, 14], [14, 11, 10, 11, 14], [14, 11, 11, 11, 14], [14, 14, 14, 14, 14]], dtype=np.float32)
    maxima = extrema.h_minima(data, 5.0)
    assert np.sum(maxima) == 0