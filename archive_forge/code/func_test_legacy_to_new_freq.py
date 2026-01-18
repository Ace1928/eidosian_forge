from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
@pytest.mark.skipif(has_pandas_ge_2_2, reason='only relevant for pandas lt 2.2')
@pytest.mark.parametrize('freq, expected', (['Y', 'YE'], ['A', 'YE'], ['Q', 'QE'], ['M', 'ME'], ['AS', 'YS'], ['YE', 'YE'], ['QE', 'QE'], ['ME', 'ME'], ['YS', 'YS']))
@pytest.mark.parametrize('n', ('', '2'))
def test_legacy_to_new_freq(freq, expected, n):
    freq = f'{n}{freq}'
    result = _legacy_to_new_freq(freq)
    expected = f'{n}{expected}'
    assert result == expected