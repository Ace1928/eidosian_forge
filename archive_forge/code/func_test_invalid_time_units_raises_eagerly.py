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
@pytest.mark.filterwarnings('ignore:Ambiguous reference date string')
def test_invalid_time_units_raises_eagerly(self) -> None:
    ds = Dataset({'time': ('time', [0, 1], {'units': 'foobar since 123'})})
    with pytest.raises(ValueError, match='unable to decode time'):
        decode_cf(ds)