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
@pytest.mark.parametrize('test', [datetime.time(), datetime.time(1, 2, 3), datetime.time(10, 12, 15, 343243)])
def test_encode_time_conversion_basic(self, test):
    output = ujson.ujson_dumps(test)
    expected = f'"{test.isoformat()}"'
    assert expected == output