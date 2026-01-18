from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.parametrize('categoricals,by', [(True, lambda df: 'b'), (False, lambda df: 'b'), (True, lambda df: df.b), (False, lambda df: df.b), (False, lambda df: df.b + 1)])
def test_groupby_get_group(categoricals, by):
    dsk = {('x', 0): pd.DataFrame({'a': [1, 2, 6], 'b': [4, 2, 7]}, index=[0, 1, 3]), ('x', 1): pd.DataFrame({'a': [4, 2, 6], 'b': [3, 3, 1]}, index=[5, 6, 8]), ('x', 2): pd.DataFrame({'a': [4, 3, 7], 'b': [1, 1, 3]}, index=[9, 9, 9])}
    if not DASK_EXPR_ENABLED:
        meta = dsk['x', 0]
        ddf = dd.DataFrame(dsk, 'x', meta, [0, 4, 9, 9])
    else:
        ddf = dd.repartition(pd.concat(dsk.values()), divisions=[0, 4, 9, 9])
    if categoricals:
        ddf = ddf.categorize(columns=['b'])
    pdf = ddf.compute()
    if PANDAS_GE_210 and categoricals and (not PANDAS_GE_300):
        ctx = pytest.warns(FutureWarning, match='The default of observed=False')
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        ddgrouped = ddf.groupby(by(ddf))
    with ctx:
        pdgrouped = pdf.groupby(by(pdf))
    assert_eq(ddgrouped.get_group(2), pdgrouped.get_group(2))
    assert_eq(ddgrouped.get_group(3), pdgrouped.get_group(3))
    assert_eq(ddgrouped.a.get_group(3), pdgrouped.a.get_group(3))
    assert_eq(ddgrouped.a.get_group(2), pdgrouped.a.get_group(2))