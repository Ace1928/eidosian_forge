import pytest
import pandas.core.dtypes.concat as _concat
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_concat_periodarray_2d():
    pi = pd.period_range('2016-01-01', periods=36, freq='D')
    arr = pi._data.reshape(6, 6)
    result = _concat.concat_compat([arr[:2], arr[2:]], axis=0)
    tm.assert_period_array_equal(result, arr)
    result = _concat.concat_compat([arr[:, :2], arr[:, 2:]], axis=1)
    tm.assert_period_array_equal(result, arr)
    msg = 'all the input array dimensions.* for the concatenation axis must match exactly'
    with pytest.raises(ValueError, match=msg):
        _concat.concat_compat([arr[:, :2], arr[:, 2:]], axis=0)
    with pytest.raises(ValueError, match=msg):
        _concat.concat_compat([arr[:2], arr[2:]], axis=1)