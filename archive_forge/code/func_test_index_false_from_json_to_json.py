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
@pytest.mark.parametrize('orient', ['split', 'table'])
@pytest.mark.parametrize('index', [True, False])
def test_index_false_from_json_to_json(self, orient, index):
    expected = DataFrame({'a': [1, 2], 'b': [3, 4]})
    dfjson = expected.to_json(orient=orient, index=index)
    result = read_json(StringIO(dfjson), orient=orient)
    tm.assert_frame_equal(result, expected)