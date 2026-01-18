from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
def testRollback2(self, dt):
    assert CBMonthEnd(10).rollback(dt) == datetime(2007, 12, 31)