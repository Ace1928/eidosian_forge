from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_groups_in_BaseGrouper():
    mi = MultiIndex.from_product([['A', 'B'], ['C', 'D']], names=['alpha', 'beta'])
    df = DataFrame({'foo': [1, 2, 1, 2], 'bar': [1, 2, 3, 4]}, index=mi)
    result = df.groupby([Grouper(level='alpha'), 'beta'])
    expected = df.groupby(['alpha', 'beta'])
    assert result.groups == expected.groups
    result = df.groupby(['beta', Grouper(level='alpha')])
    expected = df.groupby(['beta', 'alpha'])
    assert result.groups == expected.groups