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
def test_decode_null_character(self):
    wrapped_input = '"31337 \\u0000 31337"'
    assert ujson.ujson_loads(wrapped_input) == json.loads(wrapped_input)