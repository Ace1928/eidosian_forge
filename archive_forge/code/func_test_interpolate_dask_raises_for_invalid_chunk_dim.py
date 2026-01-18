from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_dask
def test_interpolate_dask_raises_for_invalid_chunk_dim():
    da, _ = make_interpolate_example_data((40, 40), 0.5)
    da = da.chunk({'time': 5})
    with pytest.raises(ValueError, match='consists of multiple chunks'):
        da.interpolate_na('time')