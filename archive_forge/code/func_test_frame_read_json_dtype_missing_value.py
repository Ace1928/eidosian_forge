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
@pytest.mark.parametrize('dtype', [True, False])
def test_frame_read_json_dtype_missing_value(self, dtype):
    result = read_json(StringIO('[null]'), dtype=dtype)
    expected = DataFrame([np.nan])
    tm.assert_frame_equal(result, expected)