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
@pytest.mark.parametrize('dropna', [True, False])
@pytest.mark.parametrize('groupby_kwargs', [pytest.param({'level': 1, 'axis': 1}, id='level_idx_axis=1'), pytest.param({'level': 1}, id='level_idx'), pytest.param({'level': [1, 'four']}, id='level_idx+name'), pytest.param({'by': 'four'}, id='level_name'), pytest.param({'by': ['one', 'two']}, id='level_name_multi_by'), pytest.param({'by': ['item0', 'one', 'two']}, id='col_name+level_name'), pytest.param({'by': ['item0']}, id='col_name'), pytest.param({'by': ['item0', 'item1']}, id='col_name_multi_by')])
def test_groupby_with_kwarg_dropna(groupby_kwargs, dropna):
    modin_df = pd.DataFrame(test_data['float_nan_data'])
    pandas_df = pandas.DataFrame(test_data['float_nan_data'])
    new_index = pandas.Index([f'item{i}' for i in range(len(pandas_df))])
    new_columns = pandas.MultiIndex.from_tuples([(i // 4, i // 2, i) for i in range(len(modin_df.columns))], names=['four', 'two', 'one'])
    modin_df.columns = new_columns
    modin_df.index = new_index
    pandas_df.columns = new_columns
    pandas_df.index = new_index
    if groupby_kwargs.get('axis', 0) == 0:
        modin_df = modin_df.T
        pandas_df = pandas_df.T
    md_grp, pd_grp = (modin_df.groupby(**groupby_kwargs, dropna=dropna), pandas_df.groupby(**groupby_kwargs, dropna=dropna))
    modin_groupby_equals_pandas(md_grp, pd_grp)
    by_kwarg = groupby_kwargs.get('by', [])
    if not (not dropna and len(by_kwarg) > 1 and any((col in modin_df.columns for col in by_kwarg))):
        df_equals(md_grp.sum(), pd_grp.sum())
        df_equals(md_grp.size(), pd_grp.size())
    if get_current_execution() != 'BaseOnPython' and any((col in modin_df.columns for col in by_kwarg)):
        df_equals(md_grp.quantile(), pd_grp.quantile())
    if not (not dropna and len(by_kwarg) > 1):
        df_equals(md_grp.first(), pd_grp.first())
        df_equals(md_grp._default_to_pandas(lambda df: df.sum()), pd_grp.sum())