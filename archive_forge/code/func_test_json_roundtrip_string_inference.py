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
def test_json_roundtrip_string_inference(orient):
    pytest.importorskip('pyarrow')
    df = DataFrame([['a', 'b'], ['c', 'd']], index=['row 1', 'row 2'], columns=['col 1', 'col 2'])
    out = df.to_json()
    with pd.option_context('future.infer_string', True):
        result = read_json(StringIO(out))
    expected = DataFrame([['a', 'b'], ['c', 'd']], dtype='string[pyarrow_numpy]', index=Index(['row 1', 'row 2'], dtype='string[pyarrow_numpy]'), columns=Index(['col 1', 'col 2'], dtype='string[pyarrow_numpy]'))
    tm.assert_frame_equal(result, expected)