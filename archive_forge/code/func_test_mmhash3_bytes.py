import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.murmurhash import murmurhash3_32
def test_mmhash3_bytes():
    assert murmurhash3_32(b'foo', 0) == -156908512
    assert murmurhash3_32(b'foo', 42) == -1322301282
    assert murmurhash3_32(b'foo', 0, positive=True) == 4138058784
    assert murmurhash3_32(b'foo', 42, positive=True) == 2972666014