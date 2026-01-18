from __future__ import annotations
from numbers import Number
import numpy as np
import pytest
import xarray as xr
from xarray.backends.api import _get_default_engine
from xarray.tests import (
def check_dataset(self, initial, final, expected_chunks):
    assert_identical(initial, final)
    assert final[self.var_name].chunks == expected_chunks