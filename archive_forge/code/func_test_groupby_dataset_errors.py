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
def test_groupby_dataset_errors() -> None:
    data = create_test_data()
    with pytest.raises(TypeError, match='`group` must be'):
        data.groupby(np.arange(10))
    with pytest.raises(ValueError, match='length does not match'):
        data.groupby(data['dim1'][:3])
    with pytest.raises(TypeError, match='`group` must be'):
        data.groupby(data.coords['dim1'].to_index())