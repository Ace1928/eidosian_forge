from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_dtypes_with_usecols(all_parsers):
    parser = all_parsers
    data = 'a,b,c\n1,2,3\n4,5,6'
    result = parser.read_csv(StringIO(data), usecols=['a', 'c'], dtype={'a': object})
    if parser.engine == 'pyarrow':
        values = [1, 4]
    else:
        values = ['1', '4']
    expected = DataFrame({'a': pd.Series(values, dtype=object), 'c': [3, 6]})
    tm.assert_frame_equal(result, expected)