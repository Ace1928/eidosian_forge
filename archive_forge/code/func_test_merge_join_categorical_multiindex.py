from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_join_categorical_multiindex():
    a = {'Cat1': Categorical(['a', 'b', 'a', 'c', 'a', 'b'], ['a', 'b', 'c']), 'Int1': [0, 1, 0, 1, 0, 0]}
    a = DataFrame(a)
    b = {'Cat': Categorical(['a', 'b', 'c', 'a', 'b', 'c'], ['a', 'b', 'c']), 'Int': [0, 0, 0, 1, 1, 1], 'Factor': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]}
    b = DataFrame(b).set_index(['Cat', 'Int'])['Factor']
    expected = merge(a, b.reset_index(), left_on=['Cat1', 'Int1'], right_on=['Cat', 'Int'], how='left')
    expected = expected.drop(['Cat', 'Int'], axis=1)
    result = a.join(b, on=['Cat1', 'Int1'])
    tm.assert_frame_equal(expected, result)
    a = {'Cat1': Categorical(['a', 'b', 'a', 'c', 'a', 'b'], ['b', 'a', 'c'], ordered=True), 'Int1': [0, 1, 0, 1, 0, 0]}
    a = DataFrame(a)
    b = {'Cat': Categorical(['a', 'b', 'c', 'a', 'b', 'c'], ['b', 'a', 'c'], ordered=True), 'Int': [0, 0, 0, 1, 1, 1], 'Factor': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]}
    b = DataFrame(b).set_index(['Cat', 'Int'])['Factor']
    expected = merge(a, b.reset_index(), left_on=['Cat1', 'Int1'], right_on=['Cat', 'Int'], how='left')
    expected = expected.drop(['Cat', 'Int'], axis=1)
    result = a.join(b, on=['Cat1', 'Int1'])
    tm.assert_frame_equal(expected, result)