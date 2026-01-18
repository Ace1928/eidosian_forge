from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouping_string_repr(self):
    mi = MultiIndex.from_arrays([list('AAB'), list('aba')])
    df = DataFrame([[1, 2, 3]], columns=mi)
    gr = df.groupby(df['A', 'a'])
    result = gr._grouper.groupings[0].__repr__()
    expected = "Grouping(('A', 'a'))"
    assert result == expected