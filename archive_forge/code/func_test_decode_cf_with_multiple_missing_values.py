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
def test_decode_cf_with_multiple_missing_values(self) -> None:
    original = Variable(['t'], [0, 1, 2], {'missing_value': np.array([0, 1])})
    expected = Variable(['t'], [np.nan, np.nan, 2], {})
    with pytest.warns(SerializationWarning, match='has multiple fill'):
        actual = conventions.decode_cf_variable('t', original)
        assert_identical(expected, actual)