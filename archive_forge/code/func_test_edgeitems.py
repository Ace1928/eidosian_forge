import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_edgeitems(self):
    np.set_printoptions(edgeitems=1, threshold=1)
    a = np.arange(27).reshape((3, 3, 3))
    assert_equal(repr(a), textwrap.dedent('            array([[[ 0, ...,  2],\n                    ...,\n                    [ 6, ...,  8]],\n\n                   ...,\n\n                   [[18, ..., 20],\n                    ...,\n                    [24, ..., 26]]])'))
    b = np.zeros((3, 3, 1, 1))
    assert_equal(repr(b), textwrap.dedent('            array([[[[0.]],\n\n                    ...,\n\n                    [[0.]]],\n\n\n                   ...,\n\n\n                   [[[0.]],\n\n                    ...,\n\n                    [[0.]]]])'))
    np.set_printoptions(legacy='1.13')
    assert_equal(repr(a), textwrap.dedent('            array([[[ 0, ...,  2],\n                    ..., \n                    [ 6, ...,  8]],\n\n                   ..., \n                   [[18, ..., 20],\n                    ..., \n                    [24, ..., 26]]])'))
    assert_equal(repr(b), textwrap.dedent('            array([[[[ 0.]],\n\n                    ..., \n                    [[ 0.]]],\n\n\n                   ..., \n                   [[[ 0.]],\n\n                    ..., \n                    [[ 0.]]]])'))