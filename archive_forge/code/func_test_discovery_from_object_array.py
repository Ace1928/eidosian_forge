import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('shape', [(), (1,)])
def test_discovery_from_object_array(self, shape):
    arr = np.array('2020-10-10', dtype=object).reshape(shape)
    res = np.array('2020-10-10', dtype='M8').reshape(shape)
    assert res.dtype == np.dtype('M8[D]')
    assert_equal(arr.astype('M8'), res)
    arr[...] = np.bytes_('2020-10-10')
    assert_equal(arr.astype('M8'), res)
    arr = arr.astype('S')
    assert_equal(arr.astype('S').astype('M8'), res)