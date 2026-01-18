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
def test_encode_array_in_array(self):
    arr_in_arr_input = [[[[]]]]
    output = ujson.ujson_dumps(arr_in_arr_input)
    assert arr_in_arr_input == json.loads(output)
    assert output == json.dumps(arr_in_arr_input)
    assert arr_in_arr_input == ujson.ujson_loads(output)