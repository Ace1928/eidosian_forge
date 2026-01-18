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
def test_encode_big_set(self):
    s = set()
    for x in range(100000):
        s.add(x)
    ujson.ujson_dumps(s)