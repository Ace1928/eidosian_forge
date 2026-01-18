from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_missing(datetime_series):
    d = datetime_series.index[0] - BDay()
    msg = "Timestamp\\('1999-12-31 00:00:00'\\)"
    with pytest.raises(KeyError, match=msg):
        datetime_series[d]