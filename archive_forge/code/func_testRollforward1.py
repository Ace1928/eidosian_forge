from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
def testRollforward1(self, dt):
    assert CBMonthEnd(10).rollforward(dt) == datetime(2008, 1, 31)