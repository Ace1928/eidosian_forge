from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
def test_asarray_tuplesafe() -> None:
    res = _asarray_tuplesafe(('a', 1))
    assert isinstance(res, np.ndarray)
    assert res.ndim == 0
    assert res.item() == ('a', 1)
    res = _asarray_tuplesafe([(0,), (1,)])
    assert res.shape == (2,)
    assert res[0] == (0,)
    assert res[1] == (1,)