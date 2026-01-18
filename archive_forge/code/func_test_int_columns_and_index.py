import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.feather_format import read_feather, to_feather  # isort:skip
def test_int_columns_and_index(self):
    df = pd.DataFrame({'a': [1, 2, 3]}, index=pd.Index([3, 4, 5], name='test'))
    self.check_round_trip(df)