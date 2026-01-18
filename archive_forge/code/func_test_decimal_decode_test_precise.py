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
def test_decimal_decode_test_precise(self):
    sut = {'a': 4.56}
    encoded = ujson.ujson_dumps(sut)
    decoded = ujson.ujson_loads(encoded, precise_float=True)
    assert sut == decoded