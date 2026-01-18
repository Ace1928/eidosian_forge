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
def test_aggregate_build_agg_args__reuse_of_intermediates():
    """Aggregate reuses intermediates. For example, with sum, count, and mean
    the sums and counts are only calculated once across the graph and reused to
    compute the mean.
    """
    from dask.dataframe.groupby import _build_agg_args
    no_mean_spec = [('foo', 'sum', 'input'), ('bar', 'count', 'input')]
    with_mean_spec = [('foo', 'sum', 'input'), ('bar', 'count', 'input'), ('baz', 'mean', 'input')]
    no_mean_chunks, no_mean_aggs, no_mean_finalizers = _build_agg_args(no_mean_spec)
    with_mean_chunks, with_mean_aggs, with_mean_finalizers = _build_agg_args(with_mean_spec)
    assert len(no_mean_chunks) == len(with_mean_chunks)
    assert len(no_mean_aggs) == len(with_mean_aggs)
    assert len(no_mean_finalizers) == len(no_mean_spec)
    assert len(with_mean_finalizers) == len(with_mean_spec)