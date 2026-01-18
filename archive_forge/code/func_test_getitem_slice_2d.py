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
def test_getitem_slice_2d(self, datetime_series):
    with pytest.raises(ValueError, match='Multi-dimensional indexing'):
        datetime_series[:, np.newaxis]