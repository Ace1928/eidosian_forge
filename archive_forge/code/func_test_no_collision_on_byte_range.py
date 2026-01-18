import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.murmurhash import murmurhash3_32
def test_no_collision_on_byte_range():
    previous_hashes = set()
    for i in range(100):
        h = murmurhash3_32(' ' * i, 0)
        assert h not in previous_hashes, 'Found collision on growing empty string'