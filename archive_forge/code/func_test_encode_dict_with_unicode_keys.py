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
@pytest.mark.parametrize('unicode_key', ['key1', 'بن'])
def test_encode_dict_with_unicode_keys(self, unicode_key):
    unicode_dict = {unicode_key: 'value1'}
    assert unicode_dict == ujson.ujson_loads(ujson.ujson_dumps(unicode_dict))