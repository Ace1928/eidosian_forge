import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_nested_array_repr(self):
    a = np.empty((2, 2), dtype=object)
    a[0, 0] = np.eye(2)
    a[0, 1] = np.eye(3)
    a[1, 0] = None
    a[1, 1] = np.ones((3, 1))
    assert_equal(repr(a), 'array([[array([[1., 0.],\n               [0., 1.]]), array([[1., 0., 0.],\n                                  [0., 1., 0.],\n                                  [0., 0., 1.]])],\n       [None, array([[1.],\n                     [1.],\n                     [1.]])]], dtype=object)')