from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
def test_cumulative_vs_cum(d) -> None:
    result = d.cumulative('z').sum()
    expected = d.cumsum('z')
    expected = expected.assign_coords(z=result['z'])
    assert_identical(result, expected)