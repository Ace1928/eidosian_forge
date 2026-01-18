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
def test_decode_coordinates(self) -> None:
    original = Dataset({'foo': ('t', [1, 2], {'coordinates': 'x'}), 'x': ('t', [4, 5])})
    actual = conventions.decode_cf(original)
    assert actual.foo.encoding['coordinates'] == 'x'