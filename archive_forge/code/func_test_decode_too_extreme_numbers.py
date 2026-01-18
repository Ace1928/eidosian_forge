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
@pytest.mark.parametrize('too_extreme_num', [f'{2 ** 64}', f'{-2 ** 63 - 1}'])
def test_decode_too_extreme_numbers(self, too_extreme_num):
    with pytest.raises(ValueError, match='Value is too big|Value is too small'):
        ujson.ujson_loads(too_extreme_num)