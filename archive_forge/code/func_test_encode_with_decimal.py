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
def test_encode_with_decimal(self):
    decimal_input = 1.0
    output = ujson.ujson_dumps(decimal_input)
    assert output == '1.0'