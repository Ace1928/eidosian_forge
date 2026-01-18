import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..affines import from_matvec, to_matvec
from ..orientations import (
from ..testing import deprecated_to, expires
def test_aff2axcodes():
    assert aff2axcodes(np.eye(4)) == tuple('RAS')
    aff = [[0, 1, 0, 10], [-1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]
    assert aff2axcodes(aff, (('L', 'R'), ('B', 'F'), ('D', 'U'))) == ('B', 'R', 'U')
    assert aff2axcodes(aff, (('L', 'R'), ('B', 'F'), ('D', 'U'))) == ('B', 'R', 'U')