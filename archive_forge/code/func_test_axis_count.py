import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import pytest
from itertools import chain
@pytest.mark.parametrize('kwargs', [{}, {'count': None}, {'bitorder': 'little'}, {'bitorder': 'little', 'count': None}, {'bitorder': 'big'}, {'bitorder': 'big', 'count': None}])
def test_axis_count(self, kwargs):
    packed0 = np.packbits(self.x, axis=0)
    unpacked0 = np.unpackbits(packed0, axis=0, **kwargs)
    assert_equal(unpacked0.dtype, np.uint8)
    if kwargs.get('bitorder', 'big') == 'big':
        assert_array_equal(unpacked0, self.padded2[:-1, :self.x.shape[1]])
    else:
        assert_array_equal(unpacked0[::-1, :], self.padded2[:-1, :self.x.shape[1]])
    packed1 = np.packbits(self.x, axis=1)
    unpacked1 = np.unpackbits(packed1, axis=1, **kwargs)
    assert_equal(unpacked1.dtype, np.uint8)
    if kwargs.get('bitorder', 'big') == 'big':
        assert_array_equal(unpacked1, self.padded2[:self.x.shape[0], :-1])
    else:
        assert_array_equal(unpacked1[:, ::-1], self.padded2[:self.x.shape[0], :-1])