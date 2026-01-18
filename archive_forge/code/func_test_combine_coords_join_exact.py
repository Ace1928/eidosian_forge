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
def test_combine_coords_join_exact(self):
    objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [1], 'y': [1]})]
    with pytest.raises(ValueError, match='cannot align.*join.*exact.*'):
        combine_nested(objs, concat_dim='x', join='exact')