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
@pytest.mark.parametrize('columns', [[(False, 'a'), (False, 'b'), (False, 'c')], [(False, 'a'), (False, 'b')], [(True, 'b'), (True, 'a'), (True, 'c')], [(True, 'a'), (True, 'b')], [(True, 'c'), (False, 'a'), (False, 'b')], [(False, 'a'), (True, 'c')]])
@pytest.mark.parametrize('drop_from_original_df', [True, False])
@pytest.mark.parametrize('as_index', [True, False])
def test_mixed_columns(columns, drop_from_original_df, as_index):
    data = {'a': [1, 1, 2, 2] * 64, 'b': [11, 11, 22, 22] * 64, 'c': [111, 111, 222, 222] * 64, 'data': [1, 2, 3, 4] * 64}
    md_df, pd_df = create_test_dfs(data)
    md_df, md_by = get_external_groupers(md_df, columns, drop_from_original_df)
    pd_df, pd_by = get_external_groupers(pd_df, columns, drop_from_original_df)
    md_grp = md_df.groupby(md_by, as_index=as_index)
    pd_grp = pd_df.groupby(pd_by, as_index=as_index)
    df_equals(md_grp.size(), pd_grp.size())
    df_equals(md_grp.sum(), pd_grp.sum())
    df_equals(md_grp.apply(lambda df: df.sum()), pd_grp.apply(lambda df: df.sum()))