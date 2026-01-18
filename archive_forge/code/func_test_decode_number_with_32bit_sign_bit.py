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
@pytest.mark.parametrize('val', [3590016419, 2 ** 31, 2 ** 32, 2 ** 32 - 1])
def test_decode_number_with_32bit_sign_bit(self, val):
    doc = f'{{"id": {val}}}'
    assert ujson.ujson_loads(doc)['id'] == val