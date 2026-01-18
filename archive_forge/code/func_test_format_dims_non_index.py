from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_format_dims_non_index() -> None:
    dims, dims_with_index = ({'x': 3, 'y': 2}, ['time'])
    formatted = fh.format_dims(dims, dims_with_index)
    assert "class='xr-has-index'" not in formatted