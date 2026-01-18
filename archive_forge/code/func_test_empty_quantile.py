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
@pytest.mark.parametrize('method', [pytest.param('tdigest', marks=pytest.mark.skipif(not crick, reason='Requires crick')), 'dask'])
def test_empty_quantile(method):
    if DASK_EXPR_ENABLED:
        pytest.raises(AssertionError, match='must provide non-')
    else:
        result = d.b.quantile([], method=method)
        exp = full.b.quantile([])
        assert result.divisions == (None, None)
        assert result.name == 'b'
        assert result.compute().name == 'b'
        assert_eq(result, exp)