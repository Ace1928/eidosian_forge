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
def test_mangled(self):
    df = DataFrame({'A': [0, 1], 'B': [1, 2], 'C': [3, 4]})
    result = df.groupby('A').agg(b=('B', lambda x: 0), c=('C', lambda x: 1))
    expected = DataFrame({'b': [0, 0], 'c': [1, 1]}, index=Index([0, 1], name='A'))
    tm.assert_frame_equal(result, expected)