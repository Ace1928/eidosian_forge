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
@pytest.mark.parametrize('unicode_input', ['Räksmörgås اسامة بن محمد بن عوض بن لادن', 'æ\x97¥Ñ\x88'])
def test_encode_unicode_conversion(self, unicode_input):
    enc = ujson.ujson_dumps(unicode_input)
    dec = ujson.ujson_loads(enc)
    assert enc == json.dumps(unicode_input)
    assert dec == json.loads(enc)