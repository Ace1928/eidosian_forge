from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from xarray.core.alignment import align
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_identical, source_ndarray
def test_init_default_index(self) -> None:
    coords = Coordinates(coords={'x': [1, 2]})
    expected = Dataset(coords={'x': [1, 2]})
    assert_identical(coords.to_dataset(), expected)
    assert 'x' in coords.xindexes