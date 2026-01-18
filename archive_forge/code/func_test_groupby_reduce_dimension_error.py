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
def test_groupby_reduce_dimension_error(array) -> None:
    grouped = array.groupby('y')
    with pytest.raises(ValueError, match='cannot reduce over dimensions'):
        grouped.mean('huh')
    with pytest.raises(ValueError, match='cannot reduce over dimensions'):
        grouped.mean(('x', 'y', 'asd'))
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        assert_identical(array.mean('x'), grouped.reduce(np.mean, 'x'))
        assert_allclose(array.mean(['x', 'z']), grouped.reduce(np.mean, ['x', 'z']))
    grouped = array.groupby('y', squeeze=False)
    assert_identical(array, grouped.mean())
    assert_identical(array.mean('x'), grouped.reduce(np.mean, 'x'))
    assert_allclose(array.mean(['x', 'z']), grouped.reduce(np.mean, ['x', 'z']))