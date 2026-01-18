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
@pytest.mark.parametrize('arr', [[], [31337]])
def test_decode_array(self, arr):
    assert arr == ujson.ujson_loads(str(arr))