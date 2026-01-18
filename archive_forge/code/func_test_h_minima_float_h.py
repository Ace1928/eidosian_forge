import math
import unittest
import numpy as np
from numpy.testing import assert_equal
from pytest import raises, warns
from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema
def test_h_minima_float_h(self):
    """specific tests for h-minima float h parameter"""
    data = np.array([[4, 4, 4, 4, 4], [4, 1, 1, 1, 4], [4, 1, 0, 1, 4], [4, 1, 1, 1, 4], [4, 4, 4, 4, 4]], dtype=np.uint8)
    h_vals = np.linspace(1.0, 2.0, 100)
    failures = 0
    for h in h_vals:
        if h % 1 != 0:
            msgs = ['possible precision loss converting image']
        else:
            msgs = []
        with expected_warnings(msgs):
            minima = extrema.h_minima(data, h)
        if minima[2, 2] == 0:
            failures += 1
    assert failures == 0