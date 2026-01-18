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
def test_default_handler(self):

    class _TestObject:

        def __init__(self, val) -> None:
            self.val = val

        @property
        def recursive_attr(self):
            return _TestObject('recursive_attr')

        def __str__(self) -> str:
            return str(self.val)
    msg = 'Maximum recursion level reached'
    with pytest.raises(OverflowError, match=msg):
        ujson.ujson_dumps(_TestObject('foo'))
    assert '"foo"' == ujson.ujson_dumps(_TestObject('foo'), default_handler=str)

    def my_handler(_):
        return 'foobar'
    assert '"foobar"' == ujson.ujson_dumps(_TestObject('foo'), default_handler=my_handler)

    def my_handler_raises(_):
        raise TypeError('I raise for anything')
    with pytest.raises(TypeError, match='I raise for anything'):
        ujson.ujson_dumps(_TestObject('foo'), default_handler=my_handler_raises)

    def my_int_handler(_):
        return 42
    assert ujson.ujson_loads(ujson.ujson_dumps(_TestObject('foo'), default_handler=my_int_handler)) == 42

    def my_obj_handler(_):
        return datetime.datetime(2013, 2, 3)
    assert ujson.ujson_loads(ujson.ujson_dumps(datetime.datetime(2013, 2, 3))) == ujson.ujson_loads(ujson.ujson_dumps(_TestObject('foo'), default_handler=my_obj_handler))
    obj_list = [_TestObject('foo'), _TestObject('bar')]
    assert json.loads(json.dumps(obj_list, default=str)) == ujson.ujson_loads(ujson.ujson_dumps(obj_list, default_handler=str))