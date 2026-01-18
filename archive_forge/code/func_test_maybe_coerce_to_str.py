from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
@pytest.mark.parametrize(['a', 'b', 'expected'], [[np.array(['a']), np.array(['b']), np.array(['a', 'b'])], [np.array([1], dtype='int64'), np.array([2], dtype='int64'), pd.Index([1, 2])]])
def test_maybe_coerce_to_str(a, b, expected):
    index = pd.Index(a).append(pd.Index(b))
    actual = utils.maybe_coerce_to_str(index, [a, b])
    assert_array_equal(expected, actual)
    assert expected.dtype == actual.dtype