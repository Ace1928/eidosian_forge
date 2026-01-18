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
def test_getitem_bool_index_positional(self):
    ser = Series({True: 1, False: 0})
    msg = 'Series.__getitem__ treating keys as positions is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ser[0]
    assert result == 1