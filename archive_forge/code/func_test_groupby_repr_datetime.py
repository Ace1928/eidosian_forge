from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
@pytest.mark.parametrize('obj', [repr_da, repr_da.to_dataset(name='a')])
def test_groupby_repr_datetime(obj) -> None:
    actual = repr(obj.groupby('t.month'))
    expected = f'{obj.__class__.__name__}GroupBy'
    expected += ", grouped over 'month'"
    expected += '\n%r groups with labels ' % len(np.unique(obj.t.dt.month))
    expected += '1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.'
    assert actual == expected