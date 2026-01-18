import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time
import dateutil
import numpy as np
import pytest
import pytz
import pandas._libs.json as ujson
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('td', [Timedelta(days=366), Timedelta(days=-1), Timedelta(hours=13, minutes=5, seconds=5), Timedelta(hours=13, minutes=20, seconds=30), Timedelta(days=-1, nanoseconds=5), Timedelta(nanoseconds=1), Timedelta(microseconds=1, nanoseconds=1), Timedelta(milliseconds=1, microseconds=1, nanoseconds=1), Timedelta(milliseconds=999, microseconds=999, nanoseconds=999)])
def test_encode_timedelta_iso(self, td):
    result = ujson.ujson_dumps(td, iso_dates=True)
    expected = f'"{td.isoformat()}"'
    assert result == expected