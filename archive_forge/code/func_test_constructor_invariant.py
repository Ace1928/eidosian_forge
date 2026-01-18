from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('values', [np.array([1.0, 1.2, 1.8, np.nan]), np.array([1, 2, 3], dtype='int64'), ['a', 'b', 'c', np.nan], [pd.Period('2014-01'), pd.Period('2014-02'), NaT], [Timestamp('2014-01-01'), Timestamp('2014-01-02'), NaT], [Timestamp('2014-01-01', tz='US/Eastern'), Timestamp('2014-01-02', tz='US/Eastern'), NaT]])
def test_constructor_invariant(self, values):
    c = Categorical(values)
    c2 = Categorical(c)
    tm.assert_categorical_equal(c, c2)