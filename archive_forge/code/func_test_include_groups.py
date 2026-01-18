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
@pytest.mark.parametrize('by', [['a'], ['a', 'b']])
@pytest.mark.parametrize('as_index', [True, False])
@pytest.mark.parametrize('include_groups', [True, False])
def test_include_groups(by, as_index, include_groups):
    data = {'a': [1, 1, 2, 2] * 64, 'b': [11, 11, 22, 22] * 64, 'c': [111, 111, 222, 222] * 64, 'data': [1, 2, 3, 4] * 64}

    def func(df):
        if include_groups:
            assert len(df.columns.intersection(by)) == len(by)
        else:
            assert len(df.columns.intersection(by)) == 0
        return df.sum()
    md_df, pd_df = create_test_dfs(data)
    eval_general(md_df, pd_df, lambda df: df.groupby(by, as_index=as_index).apply(func, include_groups=include_groups))