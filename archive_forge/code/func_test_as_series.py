import numpy as np
import numpy.polynomial.polyutils as pu
from numpy.testing import (
def test_as_series(self):
    assert_raises(ValueError, pu.as_series, [[]])
    assert_raises(ValueError, pu.as_series, [[[1, 2]]])
    assert_raises(ValueError, pu.as_series, [[1], ['a']])
    types = ['i', 'd', 'O']
    for i in range(len(types)):
        for j in range(i):
            ci = np.ones(1, types[i])
            cj = np.ones(1, types[j])
            [resi, resj] = pu.as_series([ci, cj])
            assert_(resi.dtype.char == resj.dtype.char)
            assert_(resj.dtype.char == types[i])