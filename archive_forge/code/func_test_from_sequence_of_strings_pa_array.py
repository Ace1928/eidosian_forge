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
def test_from_sequence_of_strings_pa_array(self, data, request):
    pa_dtype = data.dtype.pyarrow_dtype
    if pa.types.is_time64(pa_dtype) and pa_dtype.equals('time64[ns]') and (not PY311):
        request.applymarker(pytest.mark.xfail(reason='Nanosecond time parsing not supported.'))
    elif pa_version_under11p0 and (pa.types.is_duration(pa_dtype) or pa.types.is_decimal(pa_dtype)):
        request.applymarker(pytest.mark.xfail(raises=pa.ArrowNotImplementedError, reason=f"pyarrow doesn't support parsing {pa_dtype}"))
    elif pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
        _require_timezone_database(request)
    pa_array = data._pa_array.cast(pa.string())
    result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
    tm.assert_extension_array_equal(result, data)
    pa_array = pa_array.combine_chunks()
    result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
    tm.assert_extension_array_equal(result, data)