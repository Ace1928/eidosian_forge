import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_tz_series5(self, unit, unit2):
    first = DataFrame([[datetime(2016, 1, 1)], [datetime(2016, 1, 2)]], dtype=f'M8[{unit}]')
    first[0] = first[0].dt.tz_localize('Europe/London')
    second = DataFrame([[datetime(2016, 1, 3)]], dtype=f'M8[{unit2}]')
    second[0] = second[0].dt.tz_localize('Europe/London')
    result = concat([first, second])
    exp_unit = tm.get_finest_unit(unit, unit2)
    assert result[0].dtype == f'datetime64[{exp_unit}, Europe/London]'