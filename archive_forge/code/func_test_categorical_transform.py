import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_categorical_transform():
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    df = pd.DataFrame({'x': pd.Categorical(values, categories=[1, 2, 3]), 'y': range(len(values))})
    gb = df.groupby('x', dropna=False, observed=False)
    result = gb.transform(lambda x: x.sum())
    expected = gb.transform('sum')
    tm.assert_frame_equal(result, expected)