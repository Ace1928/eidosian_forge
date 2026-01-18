from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
@pytest.fixture
def multiindex() -> xr.Dataset:
    midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('level_1', 'level_2'))
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    return xr.Dataset({}, midx_coords)