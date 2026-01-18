from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
@pytest.mark.parametrize(['supplied', 'all_'], [([..., ...], list('abc')), ([...], list('aac'))])
def test_infix_dims_errors(supplied, all_):
    with pytest.raises(ValueError):
        list(infix_dims(supplied, all_))