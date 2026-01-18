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
def test_missing_raises(self):
    df = DataFrame({'A': [0, 1], 'B': [1, 2]})
    match = re.escape("Column(s) ['C'] do not exist")
    with pytest.raises(KeyError, match=match):
        df.groupby('A').agg(c=('C', 'sum'))