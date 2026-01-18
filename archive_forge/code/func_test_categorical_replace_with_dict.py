from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('replace_dict, final_data', [({'a': 1, 'b': 1}, [[3, 3], [2, 2]]), ({'a': 1, 'b': 2}, [[3, 1], [2, 3]])])
def test_categorical_replace_with_dict(self, replace_dict, final_data):
    df = DataFrame([[1, 1], [2, 2]], columns=['a', 'b'], dtype='category')
    final_data = np.array(final_data)
    a = pd.Categorical(final_data[:, 0], categories=[3, 2])
    ex_cat = [3, 2] if replace_dict['b'] == 1 else [1, 3]
    b = pd.Categorical(final_data[:, 1], categories=ex_cat)
    expected = DataFrame({'a': a, 'b': b})
    msg2 = 'with CategoricalDtype is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        result = df.replace(replace_dict, 3)
    tm.assert_frame_equal(result, expected)
    msg = 'Attributes of DataFrame.iloc\\[:, 0\\] \\(column name=\\"a\\"\\) are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df, expected)
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        return_value = df.replace(replace_dict, 3, inplace=True)
    assert return_value is None
    tm.assert_frame_equal(df, expected)