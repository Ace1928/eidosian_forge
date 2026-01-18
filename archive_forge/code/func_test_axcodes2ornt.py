import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..affines import from_matvec, to_matvec
from ..orientations import (
from ..testing import deprecated_to, expires
def test_axcodes2ornt():
    labels = (('left', 'right'), ('back', 'front'), ('down', 'up'))
    assert_array_equal(axcodes2ornt(('right', 'front', 'up'), labels), [[0, 1], [1, 1], [2, 1]])
    assert_array_equal(axcodes2ornt(('left', 'back', 'down'), labels), [[0, -1], [1, -1], [2, -1]])
    assert_array_equal(axcodes2ornt(('down', 'back', 'left'), labels), [[2, -1], [1, -1], [0, -1]])
    assert_array_equal(axcodes2ornt(('front', 'down', 'right'), labels), [[1, 1], [2, -1], [0, 1]])
    default = np.c_[range(3), [1] * 3]
    assert_array_equal(axcodes2ornt(('R', 'A', 'S')), default)
    assert_array_equal(axcodes2ornt(('R', None, 'S')), [[0, 1], [np.nan, np.nan], [2, 1]])
    assert_array_equal(axcodes2ornt('RAS'), default)
    with pytest.raises(ValueError):
        axcodes2ornt('rAS')
    assert_array_equal(axcodes2ornt(('R', None, 'S')), [[0, 1], [np.nan, np.nan], [2, 1]])
    with pytest.raises(ValueError):
        axcodes2ornt(('R', None, 's'))
    labels = ('SD', 'BF', 'lh')
    assert_array_equal(axcodes2ornt('BlD', labels), [[1, -1], [2, -1], [0, 1]])
    with pytest.raises(ValueError):
        axcodes2ornt('blD', labels)
    for labels in [('SD', 'BF', 'lD'), ('SD', 'SF', 'lD')]:
        with pytest.raises(ValueError):
            axcodes2ornt('blD', labels)
    for axcodes, ornt in zip(ALL_AXCODES, ALL_ORNTS):
        assert_array_equal(axcodes2ornt(axcodes), ornt)