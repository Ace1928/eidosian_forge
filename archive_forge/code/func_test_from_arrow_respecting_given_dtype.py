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
def test_from_arrow_respecting_given_dtype():
    date_array = pa.array([pd.Timestamp('2019-12-31'), pd.Timestamp('2019-12-31')], type=pa.date32())
    result = date_array.to_pandas(types_mapper={pa.date32(): ArrowDtype(pa.date64())}.get)
    expected = pd.Series([pd.Timestamp('2019-12-31'), pd.Timestamp('2019-12-31')], dtype=ArrowDtype(pa.date64()))
    tm.assert_series_equal(result, expected)