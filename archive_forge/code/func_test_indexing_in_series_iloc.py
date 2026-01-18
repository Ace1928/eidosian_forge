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
def test_indexing_in_series_iloc(series, index):
    expected = 1
    assert series.iloc[0] == expected
    expected = pd.Series([1, 2], index=index[:2])
    assert series.iloc[:2].equals(expected)