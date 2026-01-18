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
def test_getitem_out_of_bounds_indexerror(self, datetime_series):
    msg = 'index \\d+ is out of bounds for axis 0 with size \\d+'
    warn_msg = 'Series.__getitem__ treating keys as positions is deprecated'
    with pytest.raises(IndexError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            datetime_series[len(datetime_series)]