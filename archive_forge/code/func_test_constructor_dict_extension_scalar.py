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
def test_constructor_dict_extension_scalar(self, ea_scalar_and_dtype):
    ea_scalar, ea_dtype = ea_scalar_and_dtype
    df = DataFrame({'a': ea_scalar}, index=[0])
    assert df['a'].dtype == ea_dtype
    expected = DataFrame(index=[0], columns=['a'], data=ea_scalar)
    tm.assert_frame_equal(df, expected)