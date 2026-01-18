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
@requires_cftime
def test_decode_cf_datetime_transition_to_invalid(self) -> None:
    from datetime import datetime
    ds = Dataset(coords={'time': [0, 266 * 365]})
    units = 'days since 2000-01-01 00:00:00'
    ds.time.attrs = dict(units=units)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'unable to decode time')
        ds_decoded = conventions.decode_cf(ds)
    expected = np.array([datetime(2000, 1, 1, 0, 0), datetime(2265, 10, 28, 0, 0)])
    assert_array_equal(ds_decoded.time.values, expected)