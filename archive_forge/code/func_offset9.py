from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
@pytest.fixture
def offset9():
    return BusinessHour(n=3, start=['09:00', '22:00'], end=['13:00', '03:00'])