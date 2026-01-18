from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@requires_dask
def test_concat_lazy(self) -> None:
    import dask.array as da
    arrays = [DataArray(da.from_array(InaccessibleArray(np.zeros((3, 3))), 3), dims=['x', 'y']) for _ in range(2)]
    combined = concat(arrays, dim='z')
    assert combined.shape == (2, 3, 3)
    assert combined.dims == ('z', 'x', 'y')