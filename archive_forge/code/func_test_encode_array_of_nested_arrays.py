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
def test_encode_array_of_nested_arrays(self):
    nested_input = [[[[]]]] * 20
    output = ujson.ujson_dumps(nested_input)
    assert nested_input == json.loads(output)
    assert nested_input == ujson.ujson_loads(output)