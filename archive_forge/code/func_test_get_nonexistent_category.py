from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_get_nonexistent_category():
    df = DataFrame({'var': ['a', 'a', 'b', 'b'], 'val': range(4)})
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby('var').apply(lambda rows: DataFrame({'var': [rows.iloc[-1]['var']], 'val': [rows.iloc[-1]['vau']]}))