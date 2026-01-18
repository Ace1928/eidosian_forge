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
def test_getitem_dataframe_raises():
    rng = list(range(10))
    ser = Series(10, index=rng)
    df = DataFrame(rng, index=rng)
    msg = 'Indexing a Series with DataFrame is not supported, use the appropriate DataFrame column'
    with pytest.raises(TypeError, match=msg):
        ser[df > 5]