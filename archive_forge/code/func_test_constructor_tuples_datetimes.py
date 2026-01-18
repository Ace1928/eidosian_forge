from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_tuples_datetimes(self):
    values = np.array([(Timestamp('2010-01-01'),), (Timestamp('2010-01-02'),), (Timestamp('2010-01-01'),), (Timestamp('2010-01-02'),), ('a', 'b')], dtype=object)[:-1]
    result = Categorical(values)
    expected = Index([(Timestamp('2010-01-01'),), (Timestamp('2010-01-02'),)], tupleize_cols=False)
    tm.assert_index_equal(result.categories, expected)