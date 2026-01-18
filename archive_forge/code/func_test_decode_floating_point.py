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
@pytest.mark.parametrize('float_number', [1.1234567893, 1.234567893, 1.34567893, 1.4567893, 1.567893, 1.67893, 1.7893, 1.893, 1.3])
@pytest.mark.parametrize('sign', [-1, 1])
def test_decode_floating_point(self, sign, float_number):
    float_number *= sign
    tm.assert_almost_equal(float_number, ujson.ujson_loads(str(float_number)), rtol=1e-15)