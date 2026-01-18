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
@pytest.mark.parametrize(('offset', 'invalid_n'), [(BaseCFTimeOffset, 1.5), (YearBegin, 1.5), (YearEnd, 1.5), (QuarterBegin, 1.5), (QuarterEnd, 1.5), (MonthBegin, 1.5), (MonthEnd, 1.5), (Tick, 1.5), (Day, 1.5), (Hour, 1.5), (Minute, 1.5), (Second, 1.5), (Millisecond, 1.5), (Microsecond, 1.5)], ids=_id_func)
def test_cftime_offset_constructor_invalid_n(offset, invalid_n):
    with pytest.raises(TypeError):
        offset(n=invalid_n)