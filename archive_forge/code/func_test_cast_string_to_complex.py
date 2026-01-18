import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_cast_string_to_complex():
    expected = pd.DataFrame(['1.0+5j', '1.5-3j'], dtype=complex)
    result = pd.DataFrame(['1.0+5j', '1.5-3j']).astype(complex)
    tm.assert_frame_equal(result, expected)