from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
def test_dict_equiv(self):
    x = {}
    x['a'] = 3
    x['b'] = np.array([1, 2, 3])
    y = {}
    y['b'] = np.array([1.0, 2.0, 3.0])
    y['a'] = 3
    assert utils.dict_equiv(x, y)
    y['b'] = [1, 2, 3]
    assert utils.dict_equiv(x, y)
    x['b'] = [1.0, 2.0, 3.0]
    assert utils.dict_equiv(x, y)
    x['c'] = None
    assert not utils.dict_equiv(x, y)
    x['c'] = np.nan
    y['c'] = np.nan
    assert utils.dict_equiv(x, y)
    x['c'] = np.inf
    y['c'] = np.inf
    assert utils.dict_equiv(x, y)
    y = dict(y)
    assert utils.dict_equiv(x, y)
    y['b'] = 3 * np.arange(3)
    assert not utils.dict_equiv(x, y)