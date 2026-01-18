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
def test_agg_python_multiindex(multiindex_dataframe_random_data):
    grouped = multiindex_dataframe_random_data.groupby(['A', 'B'])
    result = grouped.agg('mean')
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)