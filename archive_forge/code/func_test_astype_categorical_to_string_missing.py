import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_categorical_to_string_missing(self):
    df = DataFrame(['a', 'b', np.nan])
    expected = df.astype(str)
    cat = df.astype('category')
    result = cat.astype(str)
    tm.assert_frame_equal(result, expected)