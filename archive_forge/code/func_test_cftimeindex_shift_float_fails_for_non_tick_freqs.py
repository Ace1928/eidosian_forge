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
@pytest.mark.parametrize('freq', ['YS', 'YE', 'QS', 'QE', 'MS', 'ME'])
def test_cftimeindex_shift_float_fails_for_non_tick_freqs(freq) -> None:
    a = xr.cftime_range('2000', periods=3, freq='D')
    with pytest.raises(TypeError, match='unsupported operand type'):
        a.shift(2.5, freq)