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
@pytest.mark.parametrize('invalid_type_dt', [9, datetime(2011, 1, 1)])
def test_guess_datetime_format_wrong_type_inputs(invalid_type_dt):
    with pytest.raises(TypeError, match="^Argument 'dt_str' has incorrect type \\(expected str, got .*\\)$"):
        parsing.guess_datetime_format(invalid_type_dt)