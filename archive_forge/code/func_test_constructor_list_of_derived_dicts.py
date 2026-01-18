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
def test_constructor_list_of_derived_dicts(self):

    class CustomDict(dict):
        pass
    d = {'a': 1.5, 'b': 3}
    data_custom = [CustomDict(d)]
    data = [d]
    result_custom = DataFrame(data_custom)
    result = DataFrame(data)
    tm.assert_frame_equal(result, result_custom)