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
@pytest.mark.parametrize('chunksize', [1, 2])
def test_cov_corr_meta(chunksize):
    df = pd.DataFrame({'a': np.array([1, 2, 3, 4]), 'b': np.array([1.0, 2.0, 3.0, 4.0], dtype='f4'), 'c': np.array([1.0, 2.0, 3.0, 4.0])}, index=pd.Index([1, 2, 3, 4], name='myindex'))
    ddf = dd.from_pandas(df, chunksize=chunksize)
    assert_eq(ddf.corr(), df.corr())
    assert_eq(ddf.cov(), df.cov())
    assert ddf.a.cov(ddf.b)._meta.dtype == 'f8'
    assert ddf.a.corr(ddf.b)._meta.dtype == 'f8'