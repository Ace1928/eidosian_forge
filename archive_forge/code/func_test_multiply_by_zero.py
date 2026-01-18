from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
def test_multiply_by_zero(self, dt, offset1, offset2):
    assert dt - 0 * offset1 == dt
    assert dt + 0 * offset1 == dt
    assert dt - 0 * offset2 == dt
    assert dt + 0 * offset2 == dt