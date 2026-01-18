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
def assert_combined_tile_ids_equal(dict1, dict2):
    assert len(dict1) == len(dict2)
    for k, v in dict1.items():
        assert k in dict2.keys()
        assert_equal(dict1[k], dict2[k])