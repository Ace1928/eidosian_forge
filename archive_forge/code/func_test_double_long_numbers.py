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
@pytest.mark.parametrize('long_number', [-4342969734183514, -12345678901234.568, -528656961.4399388])
def test_double_long_numbers(self, long_number):
    sut = {'a': long_number}
    encoded = ujson.ujson_dumps(sut, double_precision=15)
    decoded = ujson.ujson_loads(encoded)
    assert sut == decoded