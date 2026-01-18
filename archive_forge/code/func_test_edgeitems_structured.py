import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_edgeitems_structured(self):
    np.set_printoptions(edgeitems=1, threshold=1)
    A = np.arange(5 * 2 * 3, dtype='<i8').view([('i', '<i8', (5, 2, 3))])
    reprA = "array([([[[ 0, ...,  2], [ 3, ...,  5]], ..., [[24, ..., 26], [27, ..., 29]]],)],\n      dtype=[('i', '<i8', (5, 2, 3))])"
    assert_equal(repr(A), reprA)