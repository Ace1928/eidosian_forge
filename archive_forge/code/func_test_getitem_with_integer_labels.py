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
def test_getitem_with_integer_labels():
    ser = Series(np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2)))
    inds = [0, 2, 5, 7, 8]
    arr_inds = np.array([0, 2, 5, 7, 8])
    with pytest.raises(KeyError, match='not in index'):
        ser[inds]
    with pytest.raises(KeyError, match='not in index'):
        ser[arr_inds]