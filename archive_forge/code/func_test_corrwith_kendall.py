import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_corrwith_kendall(self):
    pytest.importorskip('scipy')
    df = DataFrame(np.random.default_rng(2).random(size=(100, 3)))
    result = df.corrwith(df ** 2, method='kendall')
    expected = Series(np.ones(len(result)))
    tm.assert_series_equal(result, expected)