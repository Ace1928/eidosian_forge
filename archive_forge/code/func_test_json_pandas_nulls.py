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
def test_json_pandas_nulls(self, nulls_fixture, request):
    if isinstance(nulls_fixture, Decimal):
        mark = pytest.mark.xfail(reason='not implemented')
        request.applymarker(mark)
    result = DataFrame([[nulls_fixture]]).to_json()
    assert result == '{"0":{"0":null}}'