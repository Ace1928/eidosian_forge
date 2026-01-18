import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
def test_dayfirst_warnings_invalid_input(self):
    arr = ['31/12/2014', '03/30/2011']
    with pytest.raises(ValueError, match=f"""^time data "03/30/2011" doesn\\'t match format "%d/%m/%Y", at position 1. {PARSING_ERR_MSG}$"""):
        to_datetime(arr, dayfirst=True)