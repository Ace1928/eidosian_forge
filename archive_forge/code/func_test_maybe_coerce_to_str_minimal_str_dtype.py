from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
def test_maybe_coerce_to_str_minimal_str_dtype():
    a = np.array(['a', 'a_long_string'])
    index = pd.Index(['a'])
    actual = utils.maybe_coerce_to_str(index, [a])
    expected = np.array('a')
    assert_array_equal(expected, actual)
    assert expected.dtype == actual.dtype