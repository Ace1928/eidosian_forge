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
def test_encode_datetime_conversion(self):
    datetime_input = datetime.datetime.fromtimestamp(time.time())
    output = ujson.ujson_dumps(datetime_input, date_unit='s')
    expected = calendar.timegm(datetime_input.utctimetuple())
    assert int(expected) == json.loads(output)
    assert int(expected) == ujson.ujson_loads(output)