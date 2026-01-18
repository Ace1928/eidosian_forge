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
def test_invalid_hypercube_input(self):
    ds = create_test_data
    datasets = [[ds(0), ds(1), ds(2)], [ds(3), ds(4)]]
    with pytest.raises(ValueError, match='sub-lists do not have consistent lengths'):
        combine_nested(datasets, concat_dim=['dim1', 'dim2'])
    datasets = [[ds(0), ds(1)], [[ds(3), ds(4)]]]
    with pytest.raises(ValueError, match='sub-lists do not have consistent depths'):
        combine_nested(datasets, concat_dim=['dim1', 'dim2'])
    datasets = [[ds(0), ds(1)], [ds(3), ds(4)]]
    with pytest.raises(ValueError, match='concat_dims has length'):
        combine_nested(datasets, concat_dim=['dim1'])