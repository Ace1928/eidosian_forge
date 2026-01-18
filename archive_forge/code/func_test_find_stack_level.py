from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
def test_find_stack_level():
    assert utils.find_stack_level() == 1
    assert utils.find_stack_level(test_mode=True) == 2

    def f():
        return utils.find_stack_level(test_mode=True)
    assert f() == 3