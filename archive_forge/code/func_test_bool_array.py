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
def test_bool_array(self):
    bool_array = np.array([True, False, True, True, False, True, False, False], dtype=bool)
    output = np.array(ujson.ujson_loads(ujson.ujson_dumps(bool_array)), dtype=bool)
    tm.assert_numpy_array_equal(bool_array, output)