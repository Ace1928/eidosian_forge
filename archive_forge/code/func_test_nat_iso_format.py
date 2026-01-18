from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
@pytest.mark.parametrize('get_nat', [lambda x: NaT, lambda x: Timedelta(x), lambda x: Timestamp(x)])
def test_nat_iso_format(get_nat):
    assert get_nat('NaT').isoformat() == 'NaT'
    assert get_nat('NaT').isoformat(timespec='nanoseconds') == 'NaT'