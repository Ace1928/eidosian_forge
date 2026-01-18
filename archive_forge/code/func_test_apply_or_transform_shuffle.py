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
@pytest.mark.parametrize('grouped', [lambda df: df.groupby('A'), lambda df: df.groupby(df['A']), lambda df: df.groupby(df['A'] + 1), lambda df: df.groupby('A')['B'], lambda df: df.groupby('A')['B'], lambda df: df.groupby(df['A'])['B'], lambda df: df.groupby(df['A'] + 1)['B'], lambda df: df.B.groupby(df['A']), lambda df: df.B.groupby(df['A'] + 1), lambda df: df.groupby('A')[['B', 'C']], lambda df: df.groupby(df['A'])[['B', 'C']], lambda df: df.groupby(df['A'] + 1)[['B', 'C']]])
@pytest.mark.parametrize('func', [lambda grp: grp.apply(lambda x: x.sum(), **INCLUDE_GROUPS), lambda grp: grp.transform(lambda x: x.sum())])
def test_apply_or_transform_shuffle(grouped, func):
    pdf = pd.DataFrame({'A': [1, 2, 3, 4] * 5, 'B': np.random.randn(20), 'C': np.random.randn(20), 'D': np.random.randn(20)})
    ddf = dd.from_pandas(pdf, 3)
    with pytest.warns(UserWarning):
        assert_eq(func(grouped(pdf)), func(grouped(ddf)))