import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_tz_with_empty(self):
    result = concat([DataFrame(date_range('2000', periods=1, tz='UTC')), DataFrame()])
    expected = DataFrame(date_range('2000', periods=1, tz='UTC'))
    tm.assert_frame_equal(result, expected)