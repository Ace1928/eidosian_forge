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
def test_agg_relabel_other_raises(self):
    df = DataFrame({'A': [0, 0, 1], 'B': [1, 2, 3]})
    grouped = df.groupby('A')
    match = 'Must provide'
    with pytest.raises(TypeError, match=match):
        grouped.agg(foo=1)
    with pytest.raises(TypeError, match=match):
        grouped.agg()
    with pytest.raises(TypeError, match=match):
        grouped.agg(a=('B', 'max'), b=(1, 2, 3))