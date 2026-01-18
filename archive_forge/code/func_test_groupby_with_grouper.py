import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
@pytest.mark.parametrize('by', [pandas.Grouper(key='time_stamp', freq='3D'), [pandas.Grouper(key='time_stamp', freq='1ME'), 'count']])
def test_groupby_with_grouper(by):
    data = {'id': [i for i in range(200)], 'time_stamp': [pd.Timestamp('2000-01-02') + datetime.timedelta(days=x) for x in range(200)]}
    for i in range(200):
        data[f'count_{i}'] = [i, i + 1] * 100
    modin_df, pandas_df = create_test_dfs(data)
    eval_general(modin_df, pandas_df, lambda df: df.groupby(by).mean(), expected_exception=False)