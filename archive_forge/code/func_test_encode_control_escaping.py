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
def test_encode_control_escaping(self):
    escaped_input = '\x19'
    enc = ujson.ujson_dumps(escaped_input)
    dec = ujson.ujson_loads(enc)
    assert escaped_input == dec
    assert enc == json.dumps(escaped_input)