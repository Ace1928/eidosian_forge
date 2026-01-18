import os
import numpy as np
import pytest
from pandas.compat import (
from pandas.errors import (
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io import pytables
from pandas.io.pytables import Term
@pytest.mark.parametrize('val', [[b'E\xc9, 17', b'', b'a', b'b', b'c'], [b'E\xc9, 17', b'a', b'b', b'c'], [b'EE, 17', b'', b'a', b'b', b'c'], [b'E\xc9, 17', b'\xf8\xfc', b'a', b'b', b'c'], [b'', b'a', b'b', b'c'], [b'\xf8\xfc', b'a', b'b', b'c'], [b'A\xf8\xfc', b'', b'a', b'b', b'c'], [np.nan, b'', b'b', b'c'], [b'A\xf8\xfc', np.nan, b'', b'b', b'c']])
@pytest.mark.parametrize('dtype', ['category', object])
def test_latin_encoding(tmp_path, setup_path, dtype, val):
    enc = 'latin-1'
    nan_rep = ''
    key = 'data'
    val = [x.decode(enc) if isinstance(x, bytes) else x for x in val]
    ser = Series(val, dtype=dtype)
    store = tmp_path / setup_path
    ser.to_hdf(store, key=key, format='table', encoding=enc, nan_rep=nan_rep)
    retr = read_hdf(store, key)
    if dtype == 'category':
        if nan_rep in ser.cat.categories:
            s_nan = ser.cat.remove_categories([nan_rep])
        else:
            s_nan = ser
    else:
        s_nan = ser.replace(nan_rep, np.nan)
    tm.assert_series_equal(s_nan, retr)