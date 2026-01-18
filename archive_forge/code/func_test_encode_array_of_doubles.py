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
def test_encode_array_of_doubles(self):
    doubles_input = [31337.31337, 31337.31337, 31337.31337, 31337.31337] * 10
    output = ujson.ujson_dumps(doubles_input)
    assert doubles_input == json.loads(output)
    assert doubles_input == ujson.ujson_loads(output)