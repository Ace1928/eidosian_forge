from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('df', [DataFrame(), DataFrame(columns=[]), DataFrame(columns=['A', 'B', 'C']), DataFrame(index=[]), DataFrame(index=['A', 'B', 'C'])])
def test_drop_duplicates_empty(df):
    result = df.drop_duplicates()
    tm.assert_frame_equal(result, df)
    result = df.copy()
    result.drop_duplicates(inplace=True)
    tm.assert_frame_equal(result, df)