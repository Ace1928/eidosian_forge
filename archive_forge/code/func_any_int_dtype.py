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
@pytest.fixture(params=tm.ALL_INT_DTYPES)
def any_int_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    return request.param