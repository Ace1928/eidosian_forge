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
@given(DATETIME_NO_TZ)
@pytest.mark.parametrize('delimiter', list(' -./'))
@pytest.mark.parametrize('dayfirst', [True, False])
@pytest.mark.parametrize('date_format', ['%d %m %Y', '%m %d %Y', '%m %Y', '%Y %m %d', '%y %m %d', '%Y%m%d', '%y%m%d'])
def test_hypothesis_delimited_date(request, date_format, dayfirst, delimiter, test_datetime):
    if date_format == '%m %Y' and delimiter == '.':
        request.applymarker(pytest.mark.xfail(reason='parse_datetime_string cannot reliably tell whether e.g. %m.%Y is a float or a date'))
    date_string = test_datetime.strftime(date_format.replace(' ', delimiter))
    except_out_dateutil, result = _helper_hypothesis_delimited_date(parsing.py_parse_datetime_string, date_string, dayfirst=dayfirst)
    except_in_dateutil, expected = _helper_hypothesis_delimited_date(du_parse, date_string, default=datetime(1, 1, 1), dayfirst=dayfirst, yearfirst=False)
    assert except_out_dateutil == except_in_dateutil
    assert result == expected