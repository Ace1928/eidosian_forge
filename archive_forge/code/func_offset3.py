from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
@pytest.fixture
def offset3():
    return BusinessHour(n=-1)