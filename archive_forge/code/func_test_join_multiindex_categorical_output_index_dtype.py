import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('how, values', [('inner', [0, 1, 2]), ('outer', [0, 1, 2]), ('left', [0, 1, 2]), ('right', [0, 2, 1])])
def test_join_multiindex_categorical_output_index_dtype(how, values):
    df1 = DataFrame({'a': Categorical([0, 1, 2]), 'b': Categorical([0, 1, 2]), 'c': [0, 1, 2]}).set_index(['a', 'b'])
    df2 = DataFrame({'a': Categorical([0, 2, 1]), 'b': Categorical([0, 2, 1]), 'd': [0, 2, 1]}).set_index(['a', 'b'])
    expected = DataFrame({'a': Categorical(values), 'b': Categorical(values), 'c': values, 'd': values}).set_index(['a', 'b'])
    result = df1.join(df2, how=how)
    tm.assert_frame_equal(result, expected)