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
@pytest.mark.parametrize('data', [np.array([['ab', 'cdef', b'X'], [1, 2, 'c']], dtype=object), np.array([['x', 1], ['y', 2]], dtype='object')])
def test_infer_dtype_error_on_mixed_types(data):
    with pytest.raises(ValueError, match='unable to infer dtype on variable'):
        conventions._infer_dtype(data, 'test')