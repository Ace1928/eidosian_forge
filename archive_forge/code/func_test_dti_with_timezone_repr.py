from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_dti_with_timezone_repr(self, tzstr):
    rng = pd.date_range('4/13/2010', '5/6/2010')
    rng_eastern = rng.tz_localize(tzstr)
    rng_repr = repr(rng_eastern)
    assert '2010-04-13 00:00:00' in rng_repr