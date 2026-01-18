import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
@pytest.mark.parametrize('typ, ad', [['float', {}], ['float', {'A': 1, 'B': 'foo', 'C': 'bar'}], ['int', {}]])
def test_constructor_mixed_dtypes(self, typ, ad):
    if typ == 'int':
        dtypes = MIXED_INT_DTYPES
        arrays = [np.array(np.random.default_rng(2).random(10), dtype=d) for d in dtypes]
    elif typ == 'float':
        dtypes = MIXED_FLOAT_DTYPES
        arrays = [np.array(np.random.default_rng(2).integers(10, size=10), dtype=d) for d in dtypes]
    for d, a in zip(dtypes, arrays):
        assert a.dtype == d
    ad.update(dict(zip(dtypes, arrays)))
    df = DataFrame(ad)
    dtypes = MIXED_FLOAT_DTYPES + MIXED_INT_DTYPES
    for d in dtypes:
        if d in df:
            assert df.dtypes[d] == d