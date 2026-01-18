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
@pytest.mark.parametrize('freq, expected_freq', [('M', 'MS'), ('ME' if PANDAS_GE_220 else 'M', 'MS'), ('MS', 'MS'), ('2M', '2MS'), ('Q', 'QS'), ('Q-FEB', 'QS-FEB'), ('2Q', '2QS'), ('2Q-FEB', '2QS-FEB'), ('2QS-FEB', '2QS-FEB'), ('BQ', 'BQS'), ('2BQ', '2BQS'), ('SM', 'SMS'), ('A', 'YS' if PANDAS_GE_220 else 'AS'), ('Y', 'YS' if PANDAS_GE_220 else 'AS'), ('A-JUN', 'YS-JUN' if PANDAS_GE_220 else 'AS-JUN'), ('Y-JUN' if PANDAS_GE_220 else 'A-JUN', 'YS-JUN' if PANDAS_GE_220 else 'AS-JUN'), ('BA', 'BYS' if PANDAS_GE_220 else 'BAS'), ('2BA', '2BYS' if PANDAS_GE_220 else '2BAS'), ('BY', 'BYS' if PANDAS_GE_220 else 'BAS'), ('Y', 'YS' if PANDAS_GE_220 else 'AS'), (pd.Timedelta(seconds=1), pd.Timedelta(seconds=1))])
def test_map_freq_to_period_start(freq, expected_freq):
    if freq in ('A', 'A-JUN', 'BA', '2BA') and PANDAS_GE_300:
        return
    if PANDAS_GE_220 and freq not in ('ME', 'MS', pd.Timedelta(seconds=1), '2QS-FEB'):
        with pytest.warns(FutureWarning, match='is deprecated and will be removed in a future version'):
            new_freq = _map_freq_to_period_start(freq)
    else:
        new_freq = _map_freq_to_period_start(freq)
    assert new_freq == expected_freq