import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
@pytest.mark.parametrize('dtype', ['c', 'S1'])
def test_inout(self, dtype):
    f = getattr(self.module, self.fprefix + '_inout')
    a = np.array(list('abc'), dtype=dtype)
    f(a, 'A')
    assert_array_equal(a, np.array(list('Abc'), dtype=a.dtype))
    f(a[1:], 'B')
    assert_array_equal(a, np.array(list('ABc'), dtype=a.dtype))
    a = np.array(['abc'], dtype=dtype)
    f(a, 'A')
    assert_array_equal(a, np.array(['Abc'], dtype=a.dtype))