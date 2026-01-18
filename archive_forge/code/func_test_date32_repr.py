from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
def test_date32_repr():
    arrow_dt = pa.array([date.fromisoformat('2020-01-01')], type=pa.date32())
    ser = pd.Series(arrow_dt, dtype=ArrowDtype(arrow_dt.type))
    assert repr(ser) == '0    2020-01-01\ndtype: date32[day][pyarrow]'