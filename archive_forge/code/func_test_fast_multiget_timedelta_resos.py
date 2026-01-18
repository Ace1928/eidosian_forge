import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
def test_fast_multiget_timedelta_resos(self):
    td = Timedelta(days=1)
    mapping1 = {td: 1}
    mapping2 = {td.as_unit('s'): 1}
    oindex = Index([td * n for n in range(3)])._values.astype(object)
    expected = lib.fast_multiget(mapping1, oindex)
    result = lib.fast_multiget(mapping2, oindex)
    tm.assert_numpy_array_equal(result, expected)
    td = Timedelta(np.timedelta64(146000, 'D'))
    assert hash(td) == hash(td.as_unit('ms'))
    assert hash(td) == hash(td.as_unit('us'))
    mapping1 = {td: 1}
    mapping2 = {td.as_unit('ms'): 1}
    oindex = Index([td * n for n in range(3)])._values.astype(object)
    expected = lib.fast_multiget(mapping1, oindex)
    result = lib.fast_multiget(mapping2, oindex)
    tm.assert_numpy_array_equal(result, expected)