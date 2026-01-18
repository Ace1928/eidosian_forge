from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
def test_either_dict_or_kwargs():
    result = either_dict_or_kwargs(dict(a=1), None, 'foo')
    expected = dict(a=1)
    assert result == expected
    result = either_dict_or_kwargs(None, dict(a=1), 'foo')
    expected = dict(a=1)
    assert result == expected
    with pytest.raises(ValueError, match='foo'):
        result = either_dict_or_kwargs(dict(a=1), dict(a=1), 'foo')