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
@pytest.mark.parametrize('inf', [np.inf, -np.inf])
@pytest.mark.parametrize('dtype', [True, False])
def test_frame_infinity(self, inf, dtype):
    df = DataFrame([[1, 2], [4, 5, 6]])
    df.loc[0, 2] = inf
    data = StringIO(df.to_json())
    result = read_json(data, dtype=dtype)
    assert np.isnan(result.iloc[0, 2])