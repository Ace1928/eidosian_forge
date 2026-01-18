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
def test_encode_null_character(self):
    wrapped_input = '31337 \x00 1337'
    output = ujson.ujson_dumps(wrapped_input)
    assert wrapped_input == json.loads(output)
    assert output == json.dumps(wrapped_input)
    assert wrapped_input == ujson.ujson_loads(output)
    alone_input = '\x00'
    output = ujson.ujson_dumps(alone_input)
    assert alone_input == json.loads(output)
    assert output == json.dumps(alone_input)
    assert alone_input == ujson.ujson_loads(output)
    assert '"  \\u0000\\r\\n "' == ujson.ujson_dumps('  \x00\r\n ')