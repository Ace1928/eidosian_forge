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
@pytest.mark.parametrize('extreme_num', [9223372036854775807, -9223372036854775808])
def test_decode_extreme_numbers(self, extreme_num):
    assert extreme_num == ujson.ujson_loads(str(extreme_num))