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
def test_encode_non_c_locale(self):
    lc_category = locale.LC_NUMERIC
    for new_locale in ('it_IT.UTF-8', 'Italian_Italy'):
        if tm.can_set_locale(new_locale, lc_category):
            with tm.set_locale(new_locale, lc_category):
                assert ujson.ujson_loads(ujson.ujson_dumps(4.78e+60)) == 4.78e+60
                assert ujson.ujson_loads('4.78', precise_float=True) == 4.78
            break