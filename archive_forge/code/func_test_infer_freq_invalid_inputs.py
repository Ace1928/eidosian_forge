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
def test_infer_freq_invalid_inputs():
    with pytest.raises(ValueError, match='must contain datetime-like objects'):
        xr.infer_freq(xr.DataArray([0, 1, 2]))
    indx = xr.cftime_range('1990-02-03', periods=4, freq='MS')
    with pytest.raises(ValueError, match='must be 1D'):
        xr.infer_freq(xr.DataArray([indx, indx]))
    with pytest.raises(ValueError, match='Need at least 3 dates to infer frequency'):
        xr.infer_freq(indx[:2])
    assert xr.infer_freq(indx[np.array([0, 2, 1, 3])]) is None
    assert xr.infer_freq(indx[np.array([0, 1, 1, 2])]) is None
    assert xr.infer_freq(indx[np.array([0, 1, 3])]) is None
    indx = xr.cftime_range('1990-02-03', periods=4, freq='QS')
    assert xr.infer_freq(indx[np.array([0, 1, 3])]) is None