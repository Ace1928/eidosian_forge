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
def test_getitem_slice_float_raises(self, datetime_series):
    msg = 'cannot do slice indexing on DatetimeIndex with these indexers \\[{key}\\] of type float'
    with pytest.raises(TypeError, match=msg.format(key='4\\.0')):
        datetime_series[4.0:10.0]
    with pytest.raises(TypeError, match=msg.format(key='4\\.5')):
        datetime_series[4.5:10.0]