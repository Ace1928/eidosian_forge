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
def test__convert_base_to_offset_invalid_index():
    with pytest.raises(ValueError, match='Can only resample'):
        _convert_base_to_offset(1, '12h', pd.Index([0]))