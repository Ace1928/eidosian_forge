import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_agg_relabel_multiindex_duplicates():
    df = DataFrame({'group': ['a', 'a', 'b', 'b'], 'A': [0, 1, 2, 3], 'B': [5, 6, 7, 8]})
    df.columns = MultiIndex.from_tuples([('x', 'group'), ('y', 'A'), ('y', 'B')])
    result = df.groupby(('x', 'group')).agg(a=(('y', 'A'), 'min'), b=(('y', 'A'), 'min'))
    idx = Index(['a', 'b'], name=('x', 'group'))
    expected = DataFrame({'a': [0, 2], 'b': [0, 2]}, index=idx)
    tm.assert_frame_equal(result, expected)