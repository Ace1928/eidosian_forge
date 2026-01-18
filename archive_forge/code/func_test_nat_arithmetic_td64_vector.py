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
@pytest.mark.parametrize('op_name', ['left_plus_right', 'right_plus_left', 'left_minus_right', 'right_minus_left'])
@pytest.mark.parametrize('box', [TimedeltaIndex, Series, TimedeltaArray._from_sequence])
def test_nat_arithmetic_td64_vector(op_name, box):
    vec = box(['1 day', '2 day'], dtype='timedelta64[ns]')
    box_nat = box([NaT, NaT], dtype='timedelta64[ns]')
    tm.assert_equal(_ops[op_name](vec, NaT), box_nat)