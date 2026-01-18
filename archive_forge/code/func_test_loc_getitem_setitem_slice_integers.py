import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
def test_loc_getitem_setitem_slice_integers(self, frame_or_series):
    index = MultiIndex(levels=[[0, 1, 2], [0, 2]], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
    obj = DataFrame(np.random.default_rng(2).standard_normal((len(index), 4)), index=index, columns=['a', 'b', 'c', 'd'])
    obj = tm.get_obj(obj, frame_or_series)
    res = obj.loc[1:2]
    exp = obj.reindex(obj.index[2:])
    tm.assert_equal(res, exp)
    obj.loc[1:2] = 7
    assert (obj.loc[1:2] == 7).values.all()