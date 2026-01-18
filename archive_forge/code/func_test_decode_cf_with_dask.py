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
@requires_dask
def test_decode_cf_with_dask(self) -> None:
    import dask.array as da
    original = Dataset({'t': ('t', [0, 1, 2], {'units': 'days since 2000-01-01'}), 'foo': ('t', [0, 0, 0], {'coordinates': 'y', 'units': 'bar'}), 'bar': ('string2', [b'a', b'b']), 'baz': ('x', [b'abc'], {'_Encoding': 'utf-8'}), 'y': ('t', [5, 10, -999], {'_FillValue': -999})}).chunk()
    decoded = conventions.decode_cf(original)
    assert all((isinstance(var.data, da.Array) for name, var in decoded.variables.items() if name not in decoded.xindexes))
    assert_identical(decoded, conventions.decode_cf(original).compute())