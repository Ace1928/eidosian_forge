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
def test_decode_cf_variable_datetime64():
    variable = Variable(['time'], pd.date_range('2000', periods=2))
    decoded = conventions.decode_cf_variable('time', variable)
    assert decoded.encoding == {}
    assert_identical(decoded, variable)