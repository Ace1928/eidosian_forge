from datetime import (
import subprocess
import sys
import numpy as np
import pytest
import pandas._config.config as cf
from pandas._libs.tslibs import to_offset
from pandas import (
import pandas._testing as tm
from pandas.plotting import (
from pandas.tseries.offsets import (
@pytest.mark.parametrize('values', [[date(1677, 1, 1), date(1677, 1, 2)], [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)]])
def test_conversion_outofbounds_datetime(self, dtc, values):
    rs = dtc.convert(values, None, None)
    xp = converter.mdates.date2num(values)
    tm.assert_numpy_array_equal(rs, xp)
    rs = dtc.convert(values[0], None, None)
    xp = converter.mdates.date2num(values[0])
    assert rs == xp