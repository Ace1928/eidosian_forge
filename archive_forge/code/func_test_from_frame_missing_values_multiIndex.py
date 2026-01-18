from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_frame_missing_values_multiIndex():
    pa = pytest.importorskip('pyarrow')
    df = pd.DataFrame({'a': Series([1, 2, None], dtype='Int64'), 'b': pd.Float64Dtype().__from_arrow__(pa.array([0.2, np.nan, None]))})
    multi_indexed = MultiIndex.from_frame(df)
    expected = MultiIndex.from_arrays([Series([1, 2, None]).astype('Int64'), pd.Float64Dtype().__from_arrow__(pa.array([0.2, np.nan, None]))], names=['a', 'b'])
    tm.assert_index_equal(multi_indexed, expected)