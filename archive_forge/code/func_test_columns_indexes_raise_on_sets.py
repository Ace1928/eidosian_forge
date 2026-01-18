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
def test_columns_indexes_raise_on_sets(self):
    data = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(ValueError, match='index cannot be a set'):
        DataFrame(data, index={'a', 'b'})
    with pytest.raises(ValueError, match='columns cannot be a set'):
        DataFrame(data, columns={'a', 'b', 'c'})