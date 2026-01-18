import datetime
import decimal
import re
import numpy as np
import pytest
import pytz
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import register_extension_dtype
from pandas.arrays import (
from pandas.core.arrays import (
from pandas.tests.extension.decimal import (
@pytest.mark.parametrize('dtype_unit', ['M8[h]', 'M8[m]', 'm8[h]', 'M8[m]'])
def test_dt64_array(dtype_unit):
    dtype_var = np.dtype(dtype_unit)
    msg = "datetime64 and timedelta64 dtype resolutions other than 's', 'ms', 'us', and 'ns' are deprecated. In future releases passing unsupported resolutions will raise an exception."
    with tm.assert_produces_warning(FutureWarning, match=re.escape(msg)):
        pd.array([], dtype=dtype_var)