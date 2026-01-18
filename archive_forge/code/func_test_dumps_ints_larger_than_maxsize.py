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
@pytest.mark.parametrize('bigNum', [2 ** 64, -2 ** 63 - 1])
def test_dumps_ints_larger_than_maxsize(self, bigNum):
    encoding = ujson.ujson_dumps(bigNum)
    assert str(bigNum) == encoding
    with pytest.raises(ValueError, match='Value is too big|Value is too small'):
        assert ujson.ujson_loads(encoding) == bigNum