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
def test_infer_freq_valid_types():
    cf_indx = xr.cftime_range('2000-01-01', periods=3, freq='D')
    assert xr.infer_freq(cf_indx) == 'D'
    assert xr.infer_freq(xr.DataArray(cf_indx)) == 'D'
    pd_indx = pd.date_range('2000-01-01', periods=3, freq='D')
    assert xr.infer_freq(pd_indx) == 'D'
    assert xr.infer_freq(xr.DataArray(pd_indx)) == 'D'
    pd_td_indx = pd.timedelta_range(start='1D', periods=3, freq='D')
    assert xr.infer_freq(pd_td_indx) == 'D'
    assert xr.infer_freq(xr.DataArray(pd_td_indx)) == 'D'