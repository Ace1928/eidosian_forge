from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
def test_dict_str(self):
    x = {b'foo': b'xxx', b'bar': b'yyyy'}
    x_rec = self.encode_decode(x)
    assert_array_equal(sorted(x.values()), sorted(x_rec.values()))
    assert_array_equal([type(e) for e in sorted(x.values())], [type(e) for e in sorted(x_rec.values())])
    assert_array_equal(sorted(x.keys()), sorted(x_rec.keys()))
    assert_array_equal([type(e) for e in sorted(x.keys())], [type(e) for e in sorted(x_rec.keys())])