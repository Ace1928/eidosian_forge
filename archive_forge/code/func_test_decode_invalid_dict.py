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
@pytest.mark.parametrize('invalid_dict', ['{{{{31337}}}}', '{{{{"key":}}}}', '{{{{"key"}}}}'])
def test_decode_invalid_dict(self, invalid_dict):
    msg = "Key name of object must be 'string' when decoding 'object'|No ':' found when decoding object value|Expected object or value"
    with pytest.raises(ValueError, match=msg):
        ujson.ujson_loads(invalid_dict)