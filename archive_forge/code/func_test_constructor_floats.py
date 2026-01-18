import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('floats', [[1.1, 2.1], np.array([1.1, 2.1])])
def test_constructor_floats(self, floats):
    msg = 'PeriodIndex does not allow floating point in construction'
    with pytest.raises(TypeError, match=msg):
        PeriodIndex(floats)