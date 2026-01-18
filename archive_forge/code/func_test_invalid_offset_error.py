from __future__ import annotations
import datetime
from typing import TypedDict
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core.pdcompat import _convert_base_to_offset
from xarray.core.resample_cftime import CFTimeGrouper
@pytest.mark.parametrize('offset', ['foo', '5MS', 10])
def test_invalid_offset_error(offset) -> None:
    cftime_index = xr.cftime_range('2000', periods=5)
    da_cftime = da(cftime_index)
    with pytest.raises(ValueError, match='offset must be'):
        da_cftime.resample(time='2D', offset=offset)