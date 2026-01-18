import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype,downcast,min_max', [('int8', 'integer', [iinfo(np.int8).min, iinfo(np.int8).max]), ('int16', 'integer', [iinfo(np.int16).min, iinfo(np.int16).max]), ('int32', 'integer', [iinfo(np.int32).min, iinfo(np.int32).max]), ('int64', 'integer', [iinfo(np.int64).min, iinfo(np.int64).max]), ('uint8', 'unsigned', [iinfo(np.uint8).min, iinfo(np.uint8).max]), ('uint16', 'unsigned', [iinfo(np.uint16).min, iinfo(np.uint16).max]), ('uint32', 'unsigned', [iinfo(np.uint32).min, iinfo(np.uint32).max]), ('uint64', 'unsigned', [iinfo(np.uint64).min, iinfo(np.uint64).max]), ('int16', 'integer', [iinfo(np.int8).min, iinfo(np.int8).max + 1]), ('int32', 'integer', [iinfo(np.int16).min, iinfo(np.int16).max + 1]), ('int64', 'integer', [iinfo(np.int32).min, iinfo(np.int32).max + 1]), ('int16', 'integer', [iinfo(np.int8).min - 1, iinfo(np.int16).max]), ('int32', 'integer', [iinfo(np.int16).min - 1, iinfo(np.int32).max]), ('int64', 'integer', [iinfo(np.int32).min - 1, iinfo(np.int64).max]), ('uint16', 'unsigned', [iinfo(np.uint8).min, iinfo(np.uint8).max + 1]), ('uint32', 'unsigned', [iinfo(np.uint16).min, iinfo(np.uint16).max + 1]), ('uint64', 'unsigned', [iinfo(np.uint32).min, iinfo(np.uint32).max + 1])])
def test_downcast_limits(dtype, downcast, min_max):
    series = to_numeric(Series(min_max), downcast=downcast)
    assert series.dtype == dtype