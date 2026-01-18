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
@pytest.mark.parametrize('by, use_da', [('x', False), ('y', False), ('y', True), ('abc', False)])
@pytest.mark.parametrize('shortcut', [True, False])
@pytest.mark.parametrize('squeeze', [None, True, False])
def test_groupby_map_identity(self, by, use_da, shortcut, squeeze, recwarn) -> None:
    expected = self.da
    if use_da:
        by = expected.coords[by]

    def identity(x):
        return x
    grouped = expected.groupby(by, squeeze=squeeze)
    actual = grouped.map(identity, shortcut=shortcut)
    assert_identical(expected, actual)
    if (by.name if use_da else by) != 'abc':
        assert len(recwarn) == (1 if squeeze in [None, True] else 0)