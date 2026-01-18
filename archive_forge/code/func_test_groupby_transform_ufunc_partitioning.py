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
@pytest.mark.parametrize('npartitions', list(range(1, 10)))
@pytest.mark.parametrize('indexed', [True, False], ids=['indexed', 'not_indexed'])
def test_groupby_transform_ufunc_partitioning(npartitions, indexed):
    pdf = pd.DataFrame({'group': [1, 2, 3, 4, 5] * 20, 'value': np.random.randn(100)})
    if indexed:
        pdf = pdf.set_index('group')
    ddf = dd.from_pandas(pdf, npartitions)
    with pytest.warns(UserWarning):
        assert_eq(pdf.groupby('group').transform(lambda series: series - series.mean()), ddf.groupby('group').transform(lambda series: series - series.mean()))
        assert_eq(pdf.groupby('group')['value'].transform(lambda series: series - series.mean()), ddf.groupby('group')['value'].transform(lambda series: series - series.mean()))