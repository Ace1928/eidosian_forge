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
@pytest.mark.parametrize('data', [{'x': [1.5, 2.5, 3.5], 'y': [9.2, 10.5, 11.8]}, {'x': [1, 2, 0], 'y': [9.2, 10.5, 11.8]}, {'x': np.array([True, True, False]), 'y': np.array([1, 2, 0]), 'z': np.array([9.2, 10.5, 11.8])}])
def test_mixed_data(data):
    df = pd.DataFrame(data)
    df2 = df.__dataframe__()
    for col_name in df.columns:
        assert df2.get_column_by_name(col_name).null_count == 0