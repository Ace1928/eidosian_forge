from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
@pytest.mark.parametrize(['supplied', 'all_', 'expected'], [(list('abc'), list('abc'), list('abc')), (['a', ..., 'c'], list('abc'), list('abc')), (['a', ...], list('abc'), list('abc')), (['c', ...], list('abc'), list('cab')), ([..., 'b'], list('abc'), list('acb')), ([...], list('abc'), list('abc'))])
def test_infix_dims(supplied, all_, expected):
    result = list(infix_dims(supplied, all_))
    assert result == expected