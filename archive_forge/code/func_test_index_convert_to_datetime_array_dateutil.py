from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_index_convert_to_datetime_array_dateutil(self):

    def _check_rng(rng):
        converted = rng.to_pydatetime()
        assert isinstance(converted, np.ndarray)
        for x, stamp in zip(converted, rng):
            assert isinstance(x, datetime)
            assert x == stamp.to_pydatetime()
            assert x.tzinfo == stamp.tzinfo
    rng = date_range('20090415', '20090519')
    rng_eastern = date_range('20090415', '20090519', tz='dateutil/US/Eastern')
    rng_utc = date_range('20090415', '20090519', tz=dateutil.tz.tzutc())
    _check_rng(rng)
    _check_rng(rng_eastern)
    _check_rng(rng_utc)