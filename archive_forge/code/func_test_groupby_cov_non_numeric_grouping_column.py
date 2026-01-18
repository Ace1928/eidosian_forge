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
def test_groupby_cov_non_numeric_grouping_column():
    pdf = pd.DataFrame({'a': 1, 'b': [pd.Timestamp('2019-12-31'), pd.Timestamp('2019-12-31'), pd.Timestamp('2019-12-31')], 'c': 2})
    ddf = dd.from_pandas(pdf, npartitions=2)
    if DASK_EXPR_ENABLED:
        assert_eq(ddf.groupby('b').cov(numeric_only=True), pdf.groupby('b').cov())
    else:
        assert_eq(ddf.groupby('b').cov(), pdf.groupby('b').cov())