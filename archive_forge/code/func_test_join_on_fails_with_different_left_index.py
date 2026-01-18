import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_on_fails_with_different_left_index(self):
    df = DataFrame({'a': np.random.default_rng(2).choice(['m', 'f'], size=3), 'b': np.random.default_rng(2).standard_normal(3)}, index=MultiIndex.from_arrays([range(3), list('abc')]))
    df2 = DataFrame({'a': np.random.default_rng(2).choice(['m', 'f'], size=10), 'b': np.random.default_rng(2).standard_normal(10)})
    msg = 'len\\(right_on\\) must equal the number of levels in the index of "left"'
    with pytest.raises(ValueError, match=msg):
        merge(df, df2, right_on='b', left_index=True)