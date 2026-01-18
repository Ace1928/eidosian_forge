import builtins
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype1,dtype2,exp_col1,exp_col2', [('float', 'Float64', np.array([True], dtype=bool), pd.array([pd.NA], dtype='boolean')), ('Int64', 'float', pd.array([pd.NA], dtype='boolean'), np.array([True], dtype=bool)), ('Int64', 'Int64', pd.array([pd.NA], dtype='boolean'), pd.array([pd.NA], dtype='boolean')), ('Float64', 'boolean', pd.array([pd.NA], dtype='boolean'), pd.array([pd.NA], dtype='boolean'))])
def test_masked_mixed_types(dtype1, dtype2, exp_col1, exp_col2):
    data = [1.0, np.nan]
    df = DataFrame({'col1': pd.array(data, dtype=dtype1), 'col2': pd.array(data, dtype=dtype2)})
    result = df.groupby([1, 1]).agg('all', skipna=False)
    expected = DataFrame({'col1': exp_col1, 'col2': exp_col2}, index=np.array([1]))
    tm.assert_frame_equal(result, expected)