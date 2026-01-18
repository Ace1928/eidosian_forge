from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_indexer_between_time(self):
    rng = date_range('1/1/2000', '1/5/2000', freq='5min')
    msg = 'Cannot convert arg \\[datetime\\.datetime\\(2010, 1, 2, 1, 0\\)\\] to a time'
    with pytest.raises(ValueError, match=msg):
        rng.indexer_between_time(datetime(2010, 1, 2, 1), datetime(2010, 1, 2, 5))