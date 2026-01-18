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
def test_encode_list_conversion(self):
    list_input = [1, 2, 3, 4]
    output = ujson.ujson_dumps(list_input)
    assert list_input == json.loads(output)
    assert list_input == ujson.ujson_loads(output)