from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
def test_describe_empty_tdigest():
    pytest.importorskip('crick')
    df_none = pd.DataFrame({'A': [None, None]})
    ddf_none = dd.from_pandas(df_none, 2)
    df_len0 = pd.DataFrame({'A': []})
    ddf_len0 = dd.from_pandas(df_len0, 2)
    ddf_nocols = dd.from_pandas(pd.DataFrame({}), 2)
    assert_eq(df_none.describe(), ddf_none.describe(percentiles_method='tdigest').compute())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        assert_eq(df_len0.describe(), ddf_len0.describe(percentiles_method='tdigest'))
        assert_eq(df_len0.describe(), ddf_len0.describe(percentiles_method='tdigest'))
    with pytest.raises(ValueError):
        ddf_nocols.describe(percentiles_method='tdigest').compute()