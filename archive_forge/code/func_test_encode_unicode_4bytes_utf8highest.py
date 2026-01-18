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
def test_encode_unicode_4bytes_utf8highest(self):
    four_bytes_input = 'ó¿¿¿TRAILINGNORMAL'
    enc = ujson.ujson_dumps(four_bytes_input)
    dec = ujson.ujson_loads(enc)
    assert enc == json.dumps(four_bytes_input)
    assert dec == json.loads(enc)