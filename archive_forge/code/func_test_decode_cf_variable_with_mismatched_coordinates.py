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
def test_decode_cf_variable_with_mismatched_coordinates() -> None:
    zeros1 = np.zeros((1, 5, 3))
    orig = Dataset({'XLONG': (['x', 'y'], zeros1.squeeze(0), {}), 'XLAT': (['x', 'y'], zeros1.squeeze(0), {}), 'foo': (['time', 'x', 'y'], zeros1, {'coordinates': 'XTIME XLONG XLAT'}), 'time': ('time', [0.0], {'units': 'hours since 2017-01-01'})})
    decoded = conventions.decode_cf(orig, decode_coords=True)
    assert decoded['foo'].encoding['coordinates'] == 'XTIME XLONG XLAT'
    assert list(decoded.coords.keys()) == ['XLONG', 'XLAT', 'time']
    decoded = conventions.decode_cf(orig, decode_coords=False)
    assert 'coordinates' not in decoded['foo'].encoding
    assert decoded['foo'].attrs.get('coordinates') == 'XTIME XLONG XLAT'
    assert list(decoded.coords.keys()) == ['time']