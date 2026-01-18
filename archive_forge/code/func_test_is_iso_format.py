from datetime import datetime
import re
from dateutil.parser import parse as du_parse
from dateutil.tz import tzlocal
from hypothesis import given
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso
from pandas.compat import (
import pandas.util._test_decorators as td
import pandas._testing as tm
from pandas._testing._hypothesis import DATETIME_NO_TZ
@pytest.mark.parametrize('fmt,expected', [('%Y %m %d %H:%M:%S', True), ('%Y/%m/%d %H:%M:%S', True), ('%Y\\%m\\%d %H:%M:%S', True), ('%Y-%m-%d %H:%M:%S', True), ('%Y.%m.%d %H:%M:%S', True), ('%Y%m%d %H:%M:%S', True), ('%Y-%m-%dT%H:%M:%S', True), ('%Y-%m-%dT%H:%M:%S%z', True), ('%Y-%m-%dT%H:%M:%S%Z', False), ('%Y-%m-%dT%H:%M:%S.%f', True), ('%Y-%m-%dT%H:%M:%S.%f%z', True), ('%Y-%m-%dT%H:%M:%S.%f%Z', False), ('%Y%m%d', True), ('%Y%m', False), ('%Y', True), ('%Y-%m-%d', True), ('%Y-%m', True)])
def test_is_iso_format(fmt, expected):
    result = strptime._test_format_is_iso(fmt)
    assert result == expected