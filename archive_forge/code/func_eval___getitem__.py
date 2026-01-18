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
def eval___getitem__(md_grp, pd_grp, item, expected_exception=None):
    eval_general(md_grp, pd_grp, lambda grp: grp[item].mean(), comparator=build_types_asserter(df_equals), expected_exception=expected_exception)
    eval_general(md_grp, pd_grp, lambda grp: grp[item].count(), comparator=build_types_asserter(df_equals), expected_exception=expected_exception)

    def build_list_agg(fns):

        def test(grp):
            res = grp[item].agg(fns)
            if res.ndim == 2:
                new_axis = fns
                if 'index' in res.columns:
                    new_axis = ['index'] + new_axis
                res = res.set_axis(new_axis, axis=1)
            return res
        return test
    eval_general(md_grp, pd_grp, build_list_agg(['mean']), comparator=build_types_asserter(df_equals), expected_exception=expected_exception)
    eval_general(md_grp, pd_grp, build_list_agg(['mean', 'count']), comparator=build_types_asserter(df_equals), expected_exception=expected_exception)
    eval_general(md_grp, pd_grp, lambda grp: grp[item].sum() if not isinstance(grp, pd.groupby.DataFrameGroupBy) else grp[item]._default_to_pandas(lambda df: df.sum()), comparator=build_types_asserter(df_equals), expected_exception=expected_exception)