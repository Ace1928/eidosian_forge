import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('unit', ['Y', 'M', 'W', 'D', 'h', 'm'])
def test_astype_from_object_to_timedelta_unit(self, unit):
    vals = [['1 Day', '2 Days', '3 Days'], ['4 Days', '5 Days', '6 Days']]
    df = DataFrame(vals, dtype=object)
    msg = "Cannot convert from timedelta64\\[ns\\] to timedelta64\\[.*\\]. Supported resolutions are 's', 'ms', 'us', 'ns'"
    with pytest.raises(ValueError, match=msg):
        df.astype(f'm8[{unit}]')