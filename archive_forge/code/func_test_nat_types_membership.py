from __future__ import annotations
import numpy as np
import pytest
from xarray.core import dtypes
def test_nat_types_membership() -> None:
    assert np.datetime64('NaT').dtype in dtypes.NAT_TYPES
    assert np.timedelta64('NaT').dtype in dtypes.NAT_TYPES
    assert np.float64 not in dtypes.NAT_TYPES