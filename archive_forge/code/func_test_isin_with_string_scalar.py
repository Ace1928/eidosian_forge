import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isin_with_string_scalar(self):
    df = DataFrame({'vals': [1, 2, 3, 4], 'ids': ['a', 'b', 'f', 'n'], 'ids2': ['a', 'n', 'c', 'n']}, index=['foo', 'bar', 'baz', 'qux'])
    msg = "only list-like or dict-like objects are allowed to be passed to DataFrame.isin\\(\\), you passed a 'str'"
    with pytest.raises(TypeError, match=msg):
        df.isin('a')
    with pytest.raises(TypeError, match=msg):
        df.isin('aaa')