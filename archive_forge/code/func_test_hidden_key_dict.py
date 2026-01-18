from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
def test_hidden_key_dict():
    hidden_key = '_hidden_key'
    data = {'a': 1, 'b': 2, hidden_key: 3}
    data_expected = {'a': 1, 'b': 2}
    hkd = utils.HiddenKeyDict(data, [hidden_key])
    assert len(hkd) == 2
    assert hidden_key not in hkd
    for k, v in data_expected.items():
        assert hkd[k] == v
    with pytest.raises(KeyError):
        hkd[hidden_key]
    with pytest.raises(KeyError):
        del hkd[hidden_key]