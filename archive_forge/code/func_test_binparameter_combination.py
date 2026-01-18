from numpy.testing import (
from numpy import (
import numpy as np
import pytest
def test_binparameter_combination(self):
    x = array([0, 0.09207008, 0.64575234, 0.12875982, 0.47390599, 0.59944483, 1])
    y = array([0, 0.14344267, 0.48988575, 0.30558665, 0.44700682, 0.15886423, 1])
    edges = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    H, xe, ye = histogram2d(x, y, (edges, 4))
    answer = array([[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert_array_equal(H, answer)
    assert_array_equal(ye, array([0.0, 0.25, 0.5, 0.75, 1]))
    H, xe, ye = histogram2d(x, y, (4, edges))
    answer = array([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    assert_array_equal(H, answer)
    assert_array_equal(xe, array([0.0, 0.25, 0.5, 0.75, 1]))