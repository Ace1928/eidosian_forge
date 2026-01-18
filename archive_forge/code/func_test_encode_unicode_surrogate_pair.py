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
def test_encode_unicode_surrogate_pair(self):
    surrogate_input = 'ð\x90\x8d\x86'
    enc = ujson.ujson_dumps(surrogate_input)
    dec = ujson.ujson_loads(enc)
    assert enc == json.dumps(surrogate_input)
    assert dec == json.loads(enc)