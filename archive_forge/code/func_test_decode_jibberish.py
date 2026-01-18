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
def test_decode_jibberish(self):
    jibberish = 'fdsa sda v9sa fdsa'
    msg = "Unexpected character found when decoding 'false'"
    with pytest.raises(ValueError, match=msg):
        ujson.ujson_loads(jibberish)