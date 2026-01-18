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
def test_dti_set_index_reindex_freq_with_tz(self):
    index = date_range(datetime(2015, 10, 1), datetime(2015, 10, 1, 23), freq='h', tz='US/Eastern')
    df = DataFrame(np.random.default_rng(2).standard_normal((24, 1)), columns=['a'], index=index)
    new_index = date_range(datetime(2015, 10, 2), datetime(2015, 10, 2, 23), freq='h', tz='US/Eastern')
    result = df.set_index(new_index)
    assert result.index.freq == index.freq