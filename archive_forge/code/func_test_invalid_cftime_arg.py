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
def test_invalid_cftime_arg() -> None:
    with pytest.warns(FutureWarning, match='Following pandas, the `closed` parameter is deprecated'):
        cftime_range('2000', '2001', None, 'YE', closed='left')