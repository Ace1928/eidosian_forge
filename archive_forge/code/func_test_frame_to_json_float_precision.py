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
@pytest.mark.skipif(not IS64, reason='not compliant on 32-bit, xref #15865')
@pytest.mark.parametrize('value,precision,expected_val', [(0.95, 1, 1.0), (1.95, 1, 2.0), (-1.95, 1, -2.0), (0.995, 2, 1.0), (0.9995, 3, 1.0), (0.9999999999999994, 15, 1.0)])
def test_frame_to_json_float_precision(self, value, precision, expected_val):
    df = DataFrame([{'a_float': value}])
    encoded = df.to_json(double_precision=precision)
    assert encoded == f'{{"a_float":{{"0":{expected_val}}}}}'