from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
@pytest.mark.parametrize('periods', [22, 50, 100])
def test_cftimeindex_repr_101_shorter(periods):
    index_101 = xr.cftime_range(start='2000', periods=101)
    index_periods = xr.cftime_range(start='2000', periods=periods)
    index_101_repr_str = index_101.__repr__()
    index_periods_repr_str = index_periods.__repr__()
    assert len(index_101_repr_str) < len(index_periods_repr_str)