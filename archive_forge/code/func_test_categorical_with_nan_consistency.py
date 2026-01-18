import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_categorical_with_nan_consistency():
    c = pd.Categorical.from_codes([-1, 0, 1, 2, 3, 4], categories=pd.date_range('2012-01-01', periods=5, name='B'))
    expected = hash_array(c, categorize=False)
    c = pd.Categorical.from_codes([-1, 0], categories=[pd.Timestamp('2012-01-01')])
    result = hash_array(c, categorize=False)
    assert result[0] in expected
    assert result[1] in expected