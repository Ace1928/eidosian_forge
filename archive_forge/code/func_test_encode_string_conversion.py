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
@pytest.mark.parametrize('ensure_ascii', [True, False])
def test_encode_string_conversion(self, ensure_ascii):
    string_input = 'A string \\ / \x08 \x0c \n \r \t </script> &'
    not_html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t <\\/script> &"'
    html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t \\u003c\\/script\\u003e \\u0026"'

    def helper(expected_output, **encode_kwargs):
        output = ujson.ujson_dumps(string_input, ensure_ascii=ensure_ascii, **encode_kwargs)
        assert output == expected_output
        assert string_input == json.loads(output)
        assert string_input == ujson.ujson_loads(output)
    helper(not_html_encoded)
    helper(not_html_encoded, encode_html_chars=False)
    helper(html_encoded, encode_html_chars=True)