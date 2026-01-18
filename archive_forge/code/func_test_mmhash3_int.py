import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.murmurhash import murmurhash3_32
def test_mmhash3_int():
    assert murmurhash3_32(3) == 847579505
    assert murmurhash3_32(3, seed=0) == 847579505
    assert murmurhash3_32(3, seed=42) == -1823081949
    assert murmurhash3_32(3, positive=False) == 847579505
    assert murmurhash3_32(3, seed=0, positive=False) == 847579505
    assert murmurhash3_32(3, seed=42, positive=False) == -1823081949
    assert murmurhash3_32(3, positive=True) == 847579505
    assert murmurhash3_32(3, seed=0, positive=True) == 847579505
    assert murmurhash3_32(3, seed=42, positive=True) == 2471885347