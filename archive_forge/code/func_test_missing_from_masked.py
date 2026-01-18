from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
def test_missing_from_masked():
    df = pd.DataFrame({'x': np.array([1.0, 2.0, 3.0, 4.0, 0.0]), 'y': np.array([1.5, 2.5, 3.5, 4.5, 0]), 'z': np.array([1.0, 0.0, 1.0, 1.0, 1.0])})
    rng = np.random.default_rng(2)
    dict_null = {col: rng.integers(low=0, high=len(df)) for col in df.columns}
    for col, num_nulls in dict_null.items():
        null_idx = df.index[rng.choice(np.arange(len(df)), size=num_nulls, replace=False)]
        df.loc[null_idx, col] = None
    df2 = df.__dataframe__()
    assert df2.get_column_by_name('x').null_count == dict_null['x']
    assert df2.get_column_by_name('y').null_count == dict_null['y']
    assert df2.get_column_by_name('z').null_count == dict_null['z']