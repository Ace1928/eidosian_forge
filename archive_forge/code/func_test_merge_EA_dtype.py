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
@pytest.mark.parametrize('expected_data, how', [([1, 2], 'outer'), ([], 'inner'), ([2], 'right'), ([1], 'left')])
def test_merge_EA_dtype(self, any_numeric_ea_dtype, how, expected_data):
    d1 = DataFrame([(1,)], columns=['id'], dtype=any_numeric_ea_dtype)
    d2 = DataFrame([(2,)], columns=['id'], dtype=any_numeric_ea_dtype)
    result = merge(d1, d2, how=how)
    exp_index = RangeIndex(len(expected_data))
    expected = DataFrame(expected_data, index=exp_index, columns=['id'], dtype=any_numeric_ea_dtype)
    tm.assert_frame_equal(result, expected)