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
def test_decode_from_unicode(self):
    unicode_input = '{"obj": 31337}'
    dec1 = ujson.ujson_loads(unicode_input)
    dec2 = ujson.ujson_loads(str(unicode_input))
    assert dec1 == dec2