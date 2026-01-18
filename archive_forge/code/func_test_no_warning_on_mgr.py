import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_no_warning_on_mgr(self):
    df = tm.SubclassedDataFrame({'X': [1, 2, 3], 'Y': [1, 2, 3]}, index=['a', 'b', 'c'])
    with tm.assert_produces_warning(None):
        df.isna()
        df['X'].isna()