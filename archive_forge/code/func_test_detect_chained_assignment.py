import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_detect_chained_assignment(using_copy_on_write, warn_copy_on_write):
    a = [12, 23]
    b = [123, None]
    c = [1234, 2345]
    d = [12345, 23456]
    tuples = [('eyes', 'left'), ('eyes', 'right'), ('ears', 'left'), ('ears', 'right')]
    events = {('eyes', 'left'): a, ('eyes', 'right'): b, ('ears', 'left'): c, ('ears', 'right'): d}
    multiind = MultiIndex.from_tuples(tuples, names=['part', 'side'])
    zed = DataFrame(events, index=['a', 'b'], columns=multiind)
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            zed['eyes']['right'].fillna(value=555, inplace=True)
    elif warn_copy_on_write:
        with tm.assert_produces_warning(None):
            zed['eyes']['right'].fillna(value=555, inplace=True)
    else:
        msg = 'A value is trying to be set on a copy of a slice from a DataFrame'
        with pytest.raises(SettingWithCopyError, match=msg):
            with tm.assert_produces_warning(None):
                zed['eyes']['right'].fillna(value=555, inplace=True)