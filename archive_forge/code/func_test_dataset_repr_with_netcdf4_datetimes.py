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
def test_dataset_repr_with_netcdf4_datetimes(self) -> None:
    attrs = {'units': 'days since 0001-01-01', 'calendar': 'noleap'}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'unable to decode time')
        ds = decode_cf(Dataset({'time': ('time', [0, 1], attrs)}))
        assert '(time) object' in repr(ds)
    attrs = {'units': 'days since 1900-01-01'}
    ds = decode_cf(Dataset({'time': ('time', [0, 1], attrs)}))
    assert '(time) datetime64[ns]' in repr(ds)