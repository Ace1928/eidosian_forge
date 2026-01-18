from datetime import timedelta
import sys
from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
def test_from_value_and_reso(self, unit, val):
    td = Timedelta._from_value_and_reso(val, unit)
    assert td._value == val
    assert td._creso == unit
    assert td.days == 106752