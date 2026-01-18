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
def test_encode_to_utf8(self):
    unencoded = 'æ\x97¥Ñ\x88'
    enc = ujson.ujson_dumps(unencoded, ensure_ascii=False)
    dec = ujson.ujson_loads(enc)
    assert enc == json.dumps(unencoded, ensure_ascii=False)
    assert dec == json.loads(enc)