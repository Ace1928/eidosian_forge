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
def test_groupby_multiindex_level() -> None:
    midx = pd.MultiIndex.from_product([list('abc'), [0, 1]], names=('one', 'two'))
    mda = xr.DataArray(np.random.rand(6, 3), [('x', midx), ('y', range(3))])
    groups = mda.groupby('one').groups
    assert groups == {'a': [0, 1], 'b': [2, 3], 'c': [4, 5]}