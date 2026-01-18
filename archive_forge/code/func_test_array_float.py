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
def test_array_float(self):
    dtype = np.float32
    arr = np.arange(100.202, 200.202, 1, dtype=dtype)
    arr = arr.reshape((5, 5, 4))
    arr_out = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=dtype)
    tm.assert_almost_equal(arr, arr_out)