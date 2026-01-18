from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
def test_list_mixed(self):
    x = [1.0, np.float32(3.5), np.complex128(4.25), b'foo']
    x_rec = self.encode_decode(x)
    assert_array_equal(x, x_rec)
    assert_array_equal([type(e) for e in x], [type(e) for e in x_rec])