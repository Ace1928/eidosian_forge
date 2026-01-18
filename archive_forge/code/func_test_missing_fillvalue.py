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
def test_missing_fillvalue(self) -> None:
    v = Variable(['x'], np.array([np.nan, 1, 2, 3]))
    v.encoding = {'dtype': 'int16'}
    with pytest.warns(Warning, match='floating point data as an integer'):
        conventions.encode_cf_variable(v)