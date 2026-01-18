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
@pytest.mark.parametrize('columns', [[(True, 'a'), (True, 'b'), (True, 'c')], [(True, 'a'), (True, 'b')], [(False, 'a'), (False, 'b'), (True, 'c')], [(False, 'a'), (True, 'c')], [(False, 'a'), (True, 'c'), (False, [1, 1, 2])]])
@pytest.mark.parametrize('as_index', [True, False])
def test_mixed_columns_not_from_df(columns, as_index):
    """
    Unlike the previous test, in this case the Series is not just a column from
    the original DataFrame, so you can't use a fasttrack.
    """
    data = {'a': [1, 1, 2], 'b': [11, 11, 22], 'c': [111, 111, 222]}
    groupby_kw = {'as_index': as_index}
    md_df, pd_df = create_test_dfs(data)
    (_, by_md), (_, by_pd) = map(lambda df: get_external_groupers(df, columns, add_plus_one=True), [md_df, pd_df])
    pd_grp = pd_df.groupby(by_pd, **groupby_kw)
    md_grp = md_df.groupby(by_md, **groupby_kw)
    modin_groupby_equals_pandas(md_grp, pd_grp)
    eval_general(md_grp, pd_grp, lambda grp: grp.size())
    eval_general(md_grp, pd_grp, lambda grp: grp.apply(lambda df: df.sum()))
    eval_general(md_grp, pd_grp, lambda grp: grp.first())