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
def test_dataframe_constructor_infer_multiindex(self):
    index_lists = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
    multi = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=[np.array(x) for x in index_lists])
    assert isinstance(multi.index, MultiIndex)
    assert not isinstance(multi.columns, MultiIndex)
    multi = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=index_lists)
    assert isinstance(multi.columns, MultiIndex)