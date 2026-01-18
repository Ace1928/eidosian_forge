from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
def test_compat_dict_intersection(self):
    assert {'b': 'B'} == utils.compat_dict_intersection(self.x, self.y)
    assert {} == utils.compat_dict_intersection(self.x, self.z)