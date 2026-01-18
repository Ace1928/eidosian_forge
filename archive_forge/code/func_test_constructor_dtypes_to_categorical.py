from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('vals', [[1, 2, 3], np.array([1, 2, 3], dtype=int), np.array(['2011-01-01', '2011-01-02'], dtype='datetime64[ns]'), [datetime(2011, 1, 1), datetime(2011, 1, 2)]])
def test_constructor_dtypes_to_categorical(self, vals):
    index = Index(vals, dtype='category')
    assert isinstance(index, CategoricalIndex)