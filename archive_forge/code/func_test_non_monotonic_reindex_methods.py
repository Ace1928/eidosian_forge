from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_non_monotonic_reindex_methods(self):
    dr = date_range('2013-08-01', periods=6, freq='B')
    data = np.random.default_rng(2).standard_normal((6, 1))
    df = DataFrame(data, index=dr, columns=list('A'))
    df_rev = DataFrame(data, index=dr[[3, 4, 5] + [0, 1, 2]], columns=list('A'))
    msg = 'index must be monotonic increasing or decreasing'
    with pytest.raises(ValueError, match=msg):
        df_rev.reindex(df.index, method='pad')
    with pytest.raises(ValueError, match=msg):
        df_rev.reindex(df.index, method='ffill')
    with pytest.raises(ValueError, match=msg):
        df_rev.reindex(df.index, method='bfill')
    with pytest.raises(ValueError, match=msg):
        df_rev.reindex(df.index, method='nearest')