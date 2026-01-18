import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('func', ['ffill', 'bfill'])
@pytest.mark.parametrize('vals', [[1, np.nan, 2], [Timestamp('2019-12-31'), NaT, Timestamp('2020-12-31')]])
def test_interpolate_triggers_copy(using_copy_on_write, vals, func):
    df = DataFrame({'a': vals})
    result = getattr(df, func)()
    assert not np.shares_memory(get_array(result, 'a'), get_array(df, 'a'))
    if using_copy_on_write:
        assert result._mgr._has_no_reference(0)