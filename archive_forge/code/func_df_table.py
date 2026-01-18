from collections import OrderedDict
from io import StringIO
import json
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.json._table_schema import (
@pytest.fixture
def df_table():
    return DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'c'], 'C': pd.date_range('2016-01-01', freq='d', periods=4), 'D': pd.timedelta_range('1h', periods=4, freq='min'), 'E': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'])), 'F': pd.Series(pd.Categorical(['a', 'b', 'c', 'c'], ordered=True)), 'G': [1.0, 2.0, 3, 4.0], 'H': pd.date_range('2016-01-01', freq='d', periods=4, tz='US/Central')}, index=pd.Index(range(4), name='idx'))