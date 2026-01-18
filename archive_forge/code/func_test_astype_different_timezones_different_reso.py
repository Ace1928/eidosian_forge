import pickle
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_astype_different_timezones_different_reso(using_copy_on_write):
    df = DataFrame({'a': date_range('2019-12-31', periods=5, freq='D', tz='US/Pacific')})
    result = df.astype('datetime64[ms, Europe/Berlin]')
    if using_copy_on_write:
        assert result._mgr._has_no_reference(0)
        assert not np.shares_memory(get_array(df, 'a'), get_array(result, 'a'))