import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_datetime_mask_slicing():
    dt_idx = pd.to_datetime(['2017-05-04', '2017-05-05'])
    m_idx = MultiIndex.from_product([dt_idx, dt_idx], names=['Idx1', 'Idx2'])
    df = DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 6]], index=m_idx, columns=['C1', 'C2'])
    result = df.loc[(dt_idx[0], df.index.get_level_values(1) > '2017-05-04'), 'C1']
    expected = Series([3], name='C1', index=MultiIndex.from_tuples([(pd.Timestamp('2017-05-04'), pd.Timestamp('2017-05-05'))], names=['Idx1', 'Idx2']))
    tm.assert_series_equal(result, expected)