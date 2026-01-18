import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin_not_lossy(self):
    val = 1666880195890293744
    df = DataFrame({'a': [val], 'b': [1.0]})
    result = df.isin([val])
    expected = DataFrame({'a': [True], 'b': [False]})
    tm.assert_frame_equal(result, expected)