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
def test_encode_object(self):

    class _TestObject:

        def __init__(self, a, b, _c, d) -> None:
            self.a = a
            self.b = b
            self._c = _c
            self.d = d

        def e(self):
            return 5
    test_object = _TestObject(a=1, b=2, _c=3, d=4)
    assert ujson.ujson_loads(ujson.ujson_dumps(test_object)) == {'a': 1, 'b': 2, 'd': 4}