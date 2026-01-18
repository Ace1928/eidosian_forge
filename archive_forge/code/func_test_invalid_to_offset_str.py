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
@pytest.mark.parametrize('freq', ['Z', '7min2', 'AM', 'M-', 'AS-', 'QS-', '1H1min'])
def test_invalid_to_offset_str(freq):
    with pytest.raises(ValueError):
        to_offset(freq)