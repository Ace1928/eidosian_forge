from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_construct_with_weeks_unit_overflow():
    with pytest.raises(OutOfBoundsTimedelta, match='without overflow'):
        Timedelta(1000000000000000000, unit='W')
    with pytest.raises(OutOfBoundsTimedelta, match='without overflow'):
        Timedelta(1e+18, unit='W')