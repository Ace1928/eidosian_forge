import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
def test_h_maxima_float_h(self):
    """specific tests for h-maxima float h parameter"""
    data = np.array([[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 3, 4, 3, 0], [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]], dtype=np.uint8)
    h_vals = np.linspace(1.0, 2.0, 100)
    failures = 0
    for h in h_vals:
        if h % 1 != 0:
            msgs = ['possible precision loss converting image']
        else:
            msgs = []
        with expected_warnings(msgs):
            maxima = extrema.h_maxima(data, h)
        if maxima[2, 2] == 0:
            failures += 1
    assert failures == 0