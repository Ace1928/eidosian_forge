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
def test_getitem_out_of_bounds_empty_rangeindex_keyerror(self):
    ser = Series([], dtype=object)
    with pytest.raises(KeyError, match='-1'):
        ser[-1]