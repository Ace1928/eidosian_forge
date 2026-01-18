import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_listlike_labels(self, df):
    result = df.loc[['c', 'a']]
    expected = df.iloc[[4, 0, 1, 5]]
    tm.assert_frame_equal(result, expected, check_index_type=True)