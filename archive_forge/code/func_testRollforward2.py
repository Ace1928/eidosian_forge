from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
def testRollforward2(self, _offset):
    assert _offset(10).rollforward(datetime(2008, 1, 5)) == datetime(2008, 1, 7)