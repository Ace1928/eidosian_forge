import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_lstrip(self):
    tgt = [[b'abc ', b''], [b'12345', b'MixedCase'], [b'123 \t 345 \x00 ', b'UPPER']]
    assert_(issubclass(self.A.lstrip().dtype.type, np.bytes_))
    assert_array_equal(self.A.lstrip(), tgt)
    tgt = [[b' abc', b''], [b'2345', b'ixedCase'], [b'23 \t 345 \x00', b'UPPER']]
    assert_array_equal(self.A.lstrip([b'1', b'M']), tgt)
    tgt = [['Σ ', ''], ['12345', 'MixedCase'], ['123 \t 345 \x00 ', 'UPPER']]
    assert_(issubclass(self.B.lstrip().dtype.type, np.str_))
    assert_array_equal(self.B.lstrip(), tgt)