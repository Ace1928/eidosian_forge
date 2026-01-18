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
def test_asi8_empty_cftimeindex():
    index = xr.CFTimeIndex([])
    result = index.asi8
    expected = np.array([], dtype=np.int64)
    np.testing.assert_array_equal(result, expected)