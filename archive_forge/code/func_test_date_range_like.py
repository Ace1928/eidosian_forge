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
@requires_cftime
@pytest.mark.parametrize('start,freq,cal_src,cal_tgt,use_cftime,exp0,exp_pd', [('2020-02-01', '4ME', 'standard', 'noleap', None, '2020-02-28', False), ('2020-02-01', 'ME', 'noleap', 'gregorian', True, '2020-02-29', True), ('2020-02-01', 'QE-DEC', 'noleap', 'gregorian', True, '2020-03-31', True), ('2020-02-01', 'YS-FEB', 'noleap', 'gregorian', True, '2020-02-01', True), ('2020-02-01', 'YE-FEB', 'noleap', 'gregorian', True, '2020-02-29', True), ('2020-02-01', '-1YE-FEB', 'noleap', 'gregorian', True, '2020-02-29', True), ('2020-02-28', '3h', 'all_leap', 'gregorian', False, '2020-02-28', True), ('2020-03-30', 'ME', '360_day', 'gregorian', False, '2020-03-31', True), ('2020-03-31', 'ME', 'gregorian', '360_day', None, '2020-03-30', False), ('2020-03-31', '-1ME', 'gregorian', '360_day', None, '2020-03-30', False)])
def test_date_range_like(start, freq, cal_src, cal_tgt, use_cftime, exp0, exp_pd):
    expected_freq = freq
    source = date_range(start, periods=12, freq=freq, calendar=cal_src)
    out = date_range_like(source, cal_tgt, use_cftime=use_cftime)
    assert len(out) == 12
    assert infer_freq(out) == expected_freq
    assert out[0].isoformat().startswith(exp0)
    if exp_pd:
        assert isinstance(out, pd.DatetimeIndex)
    else:
        assert isinstance(out, CFTimeIndex)
        assert out.calendar == cal_tgt