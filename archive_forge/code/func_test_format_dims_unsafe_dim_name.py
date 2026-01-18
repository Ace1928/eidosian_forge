from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_format_dims_unsafe_dim_name() -> None:
    dims = {'<x>': 3, 'y': 2}
    dims_with_index: list = []
    formatted = fh.format_dims(dims, dims_with_index)
    assert '&lt;x&gt;' in formatted