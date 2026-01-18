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
@pytest.mark.parametrize('shape', [(10, 10), (5, 5, 4), (100, 1)])
def test_array_reshaped(self, shape):
    arr = np.arange(100)
    arr = arr.reshape(shape)
    tm.assert_numpy_array_equal(np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr)