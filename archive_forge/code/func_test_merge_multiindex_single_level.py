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
def test_merge_multiindex_single_level():
    df = DataFrame({'col': ['A', 'B']})
    df2 = DataFrame(data={'b': [100]}, index=MultiIndex.from_tuples([('A',), ('C',)], names=['col']))
    expected = DataFrame({'col': ['A', 'B'], 'b': [100, np.nan]})
    result = df.merge(df2, left_on=['col'], right_index=True, how='left')
    tm.assert_frame_equal(result, expected)