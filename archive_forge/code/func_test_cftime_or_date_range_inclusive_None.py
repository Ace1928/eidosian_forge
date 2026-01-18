from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
@pytest.mark.parametrize('function', [cftime_range, date_range])
def test_cftime_or_date_range_inclusive_None(function) -> None:
    if function == cftime_range and (not has_cftime):
        pytest.skip('requires cftime')
    result_None = function('2000-01-01', '2000-01-04')
    result_both = function('2000-01-01', '2000-01-04', inclusive='both')
    np.testing.assert_equal(result_None.values, result_both.values)