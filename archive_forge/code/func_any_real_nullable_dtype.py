from __future__ import annotations
from collections import abc
from datetime import (
from decimal import Decimal
import operator
import os
from typing import (
from dateutil.tz import (
import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest
from pytz import (
from pandas._config.config import _get_option
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.indexes.api import (
from pandas.util.version import Version
import zoneinfo
@pytest.fixture(params=tm.ALL_REAL_NULLABLE_DTYPES)
def any_real_nullable_dtype(request):
    """
    Parameterized fixture for all real dtypes that can hold NA.

    * float
    * 'float32'
    * 'float64'
    * 'Float32'
    * 'Float64'
    * 'UInt8'
    * 'UInt16'
    * 'UInt32'
    * 'UInt64'
    * 'Int8'
    * 'Int16'
    * 'Int32'
    * 'Int64'
    * 'uint8[pyarrow]'
    * 'uint16[pyarrow]'
    * 'uint32[pyarrow]'
    * 'uint64[pyarrow]'
    * 'int8[pyarrow]'
    * 'int16[pyarrow]'
    * 'int32[pyarrow]'
    * 'int64[pyarrow]'
    * 'float[pyarrow]'
    * 'double[pyarrow]'
    """
    return request.param