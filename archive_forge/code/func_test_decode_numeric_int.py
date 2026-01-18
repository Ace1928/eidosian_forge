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
@pytest.mark.parametrize('numeric_int_as_str', ['31337', '-31337'])
def test_decode_numeric_int(self, numeric_int_as_str):
    assert int(numeric_int_as_str) == ujson.ujson_loads(numeric_int_as_str)