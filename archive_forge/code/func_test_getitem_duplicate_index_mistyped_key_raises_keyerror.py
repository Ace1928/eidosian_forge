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
def test_getitem_duplicate_index_mistyped_key_raises_keyerror():
    ser = Series([2, 5, 6, 8], index=[2.0, 4.0, 4.0, 5.0])
    with pytest.raises(KeyError, match='None'):
        ser[None]
    with pytest.raises(KeyError, match='None'):
        ser.index.get_loc(None)
    with pytest.raises(KeyError, match='None'):
        ser.index._engine.get_loc(None)