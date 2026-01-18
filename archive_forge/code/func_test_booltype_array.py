from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from xarray import (
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical
from xarray.tests import (
from xarray.tests.test_backends import CFEncodedBase
def test_booltype_array(self) -> None:
    x = np.array([1, 0, 1, 1, 0], dtype='i1')
    bx = coding.variables.BoolTypeArray(x)
    assert bx.dtype == bool
    assert_array_equal(bx, np.array([True, False, True, True, False], dtype=bool))