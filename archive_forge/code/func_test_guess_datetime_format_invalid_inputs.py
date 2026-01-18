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
@pytest.mark.parametrize('invalid_dt', ['01/2013', '12:00:00', '1/1/1/1', 'this_is_not_a_datetime', '51a', '13/2019', '202001', '2020/01', '87156549591102612381000001219H5'])
def test_guess_datetime_format_invalid_inputs(invalid_dt):
    assert parsing.guess_datetime_format(invalid_dt) is None