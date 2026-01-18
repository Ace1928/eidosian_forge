import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
def test_literal_json_deprecation():
    expected = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
    jsonl = '{"a": 1, "b": 2}\n        {"a": 3, "b": 4}\n        {"a": 5, "b": 6}\n        {"a": 7, "b": 8}'
    msg = "Passing literal json to 'read_json' is deprecated and will be removed in a future version. To read from a literal string, wrap it in a 'StringIO' object."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        try:
            read_json(jsonl, lines=False)
        except ValueError:
            pass
    with tm.assert_produces_warning(FutureWarning, match=msg):
        read_json(expected.to_json(), lines=False)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=True)
        tm.assert_frame_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        try:
            result = read_json('{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n', lines=False)
        except ValueError:
            pass
    with tm.assert_produces_warning(FutureWarning, match=msg):
        try:
            result = read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=False)
        except ValueError:
            pass
        tm.assert_frame_equal(result, expected)