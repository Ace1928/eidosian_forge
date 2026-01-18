import sys
from io import BytesIO
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_
from pytest import raises as assert_raises
import scipy.io.matlab._byteordercodes as boc
import scipy.io.matlab._streams as streams
import scipy.io.matlab._mio5_params as mio5p
import scipy.io.matlab._mio5_utils as m5u
def test_byteswap():
    for val in (1, 256, 65536):
        a = np.array(val, dtype=np.uint32)
        b = a.byteswap()
        c = m5u.byteswap_u4(a)
        assert_equal(b.item(), c)
        d = m5u.byteswap_u4(c)
        assert_equal(a.item(), d)