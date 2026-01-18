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
def test_invalid_coordinates(self) -> None:
    original = Dataset({'foo': ('t', [1, 2], {'coordinates': 'invalid'})})
    decoded = Dataset({'foo': ('t', [1, 2], {}, {'coordinates': 'invalid'})})
    actual = conventions.decode_cf(original)
    assert_identical(decoded, actual)
    actual = conventions.decode_cf(original, decode_coords=False)
    assert_identical(original, actual)