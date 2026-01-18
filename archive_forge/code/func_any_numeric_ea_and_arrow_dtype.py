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
@pytest.fixture(params=tm.ALL_INT_EA_DTYPES + tm.FLOAT_EA_DTYPES + tm.ALL_INT_PYARROW_DTYPES_STR_REPR + tm.FLOAT_PYARROW_DTYPES_STR_REPR)
def any_numeric_ea_and_arrow_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype and
    any float ea dtypes.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    * 'uint8[pyarrow]'
    * 'int8[pyarrow]'
    * 'uint16[pyarrow]'
    * 'int16[pyarrow]'
    * 'uint32[pyarrow]'
    * 'int32[pyarrow]'
    * 'uint64[pyarrow]'
    * 'int64[pyarrow]'
    * 'float32[pyarrow]'
    * 'float64[pyarrow]'
    """
    return request.param