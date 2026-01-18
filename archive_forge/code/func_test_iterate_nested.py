from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
@pytest.mark.parametrize('nested_list, expected', [([], []), ([1], [1]), ([1, 2, 3], [1, 2, 3]), ([[1]], [1]), ([[1, 2], [3, 4]], [1, 2, 3, 4]), ([[[1, 2, 3], [4]], [5, 6]], [1, 2, 3, 4, 5, 6])])
def test_iterate_nested(nested_list, expected):
    assert list(iterate_nested(nested_list)) == expected