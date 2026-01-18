from __future__ import annotations
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from xarray import (
from xarray.core import dtypes
from xarray.core.combine import (
from xarray.tests import assert_equal, assert_identical, requires_cftime
from xarray.tests.test_dataset import create_test_data
def test_combine_by_coords_raises_for_differing_types():
    da_1 = DataArray([0], dims=['time'], coords=[['a']], name='a').to_dataset()
    da_2 = DataArray([1], dims=['time'], coords=[[b'b']], name='a').to_dataset()
    with pytest.raises(TypeError, match="Cannot combine along dimension 'time' with mixed types."):
        combine_by_coords([da_1, da_2])