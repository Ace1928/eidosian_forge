import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_filter_regex_non_string(self):
    df = DataFrame(np.random.default_rng(2).random((3, 2)), columns=['STRING', 123])
    result = df.filter(regex='STRING')
    expected = df[['STRING']]
    tm.assert_frame_equal(result, expected)