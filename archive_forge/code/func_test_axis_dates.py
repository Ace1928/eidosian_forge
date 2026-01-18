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
def test_axis_dates(self, datetime_series, datetime_frame):
    json = StringIO(datetime_frame.to_json())
    result = read_json(json)
    tm.assert_frame_equal(result, datetime_frame)
    json = StringIO(datetime_series.to_json())
    result = read_json(json, typ='series')
    tm.assert_series_equal(result, datetime_series, check_names=False)
    assert result.name is None